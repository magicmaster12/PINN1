"""
基于物理信息的神经网络（PINN）用于声学辐射问题
直接在 r ∈ [1, 5] 区域训练，不再做保角映射
使用拉丁超立方采样生成训练点
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.optimize import minimize
from pyDOE import lhs
from scipy.special import hankel1
import torch.autograd as autograd
import os
from lr_adaptor import LR_Adaptor  

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

plt.switch_backend('Agg')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(1234)
np.random.seed(1234)

# 参数设置
k = 1.0
omega = 2.0 * math.pi * 1.0
n_iter = 5000
n_train = 20000
radius = 1.0
bc_value = 100.0
outer_radius = 5.0

misfit = []

def complex_gradients(Y_real, Y_imag, x):
    G_real = torch.autograd.grad(Y_real.sum(), x, create_graph=True)[0]
    G_imag = torch.autograd.grad(Y_imag.sum(), x, create_graph=True)[0]
    return G_real, G_imag

class PhysicsInformedNN(torch.nn.Module):
    def __init__(self, x, z, layers, omega):
        super(PhysicsInformedNN, self).__init__()
        X = np.concatenate([x, z], 1)
        self.lb = torch.tensor(X.min(0), dtype=torch.float32)
        self.ub = torch.tensor(X.max(0), dtype=torch.float32)
        self.X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        self.x = torch.tensor(X[:,0:1], dtype=torch.float32, requires_grad=True)
        self.z = torch.tensor(X[:,1:2], dtype=torch.float32, requires_grad=True)
        self.layers = layers
        self.omega = omega
        
        self.net = self.initialize_NN(layers)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.x = self.x.to(self.device)
        self.z = self.z.to(self.device)
        
        # Initialize loss_weights as a tensor on self.device
        self.loss_weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float32, device=self.device)
        
        self.optimizer_adam = torch.optim.Adam(self.parameters(), lr=0.001)
        self.train_vars = list(self.parameters())
        self.sizes = [p.numel() for p in self.train_vars]
        self.total_size = sum(self.sizes)

    def initialize_NN(self, layers):
        modules = []
        for l in range(len(layers)-1):
            modules.append(torch.nn.Linear(layers[l], layers[l+1]))
            if l < len(layers)-2:
                modules.append(torch.nn.Tanh())
        return torch.nn.Sequential(*modules)
    
    def xavier_init(self):
        for module in self.net:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, X):
        H = 2.0 * (X - self.lb.to(X.device)) / (self.ub.to(X.device) - self.lb.to(X.device)) - 1.0
        return self.net(H)
    
    def net_helmholtz(self, x, z):
        X = torch.cat([x, z], dim=1)
        u_pred = self.forward(X)
        u_real = u_pred[:,0:1]
        u_imag = u_pred[:,1:2]

        dudx_real, dudx_imag = complex_gradients(u_real, u_imag, x)
        dudz_real, dudz_imag = complex_gradients(u_real, u_imag, z)
        dudxx_real, dudxx_imag = complex_gradients(dudx_real, dudx_imag, x)
        dudzz_real, dudzz_imag = complex_gradients(dudz_real, dudz_imag, z)

        f_real = dudxx_real + dudzz_real + k**2 * u_real
        f_imag = dudxx_imag + dudzz_imag + k**2 * u_imag
        f_loss = f_real**2 + f_imag**2
        return u_real, u_imag, f_loss

    def compute_loss(self, x, z):
        """计算总损失（PDE + 内边界 + 外边界 Sommerfeld），使用 self.loss_weights 加权"""
        u_real_pred, u_imag_pred, f_loss = self.net_helmholtz(x, z)

        # PDE 损失
        loss_pde = torch.mean(f_loss)

        # 内边界 r=1
        r_mapped = torch.sqrt(x**2 + z**2)
        bc_inner_mask = (torch.abs(r_mapped - radius) < 0.01).float()
        loss_bc_inner = torch.mean(bc_inner_mask * ((u_real_pred - bc_value)**2 + u_imag_pred**2))

        # 外边界 r=outer_radius，Sommerfeld 条件
        bc_outer_mask = (torch.abs(r_mapped - outer_radius) < 0.01).float()
        dudx_real, dudx_imag = complex_gradients(u_real_pred, u_imag_pred, x)
        dudz_real, dudz_imag = complex_gradients(u_real_pred, u_imag_pred, z)
        dudr_real = (x * dudx_real + z * dudz_real) / r_mapped
        dudr_imag = (x * dudx_imag + z * dudz_imag) / r_mapped
        loss_bc_outer = torch.mean(bc_outer_mask * ((dudr_real + k * u_imag_pred)**2 + (dudr_imag - k * u_real_pred)**2))

        # 总损失，使用 self.loss_weights 加权
        loss_total = (self.loss_weights[0] * loss_pde +
                      self.loss_weights[1] * loss_bc_inner +
                      self.loss_weights[2] * loss_bc_outer)
        
        return loss_total, loss_pde, loss_bc_inner, loss_bc_outer

    def get_loss_and_grad(self, theta):
        start = 0
        for param, size in zip(self.parameters(), self.sizes):
            param.data = torch.tensor(theta[start:start+size], dtype=torch.float32, device=self.device).reshape(param.shape)
            start += size
        self.zero_grad()
        loss_total, loss_pde, loss_bc_inner, loss_bc_outer = self.compute_loss(self.x, self.z)
        loss_total.backward()
        grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=self.device) 
                         for p in self.parameters()])
        return loss_total.item(), grad.cpu().numpy().astype(np.float64), loss_pde.item(), loss_bc_inner.item(), loss_bc_outer.item()

    def train_lbfgs(self, maxiter=1000):
        def callback(theta):
            loss, _, pde, bc_inner, bc_outer = self.get_loss_and_grad(theta)
            misfit.append(loss)
            print(f'L-BFGS 迭代: {len(misfit)}, 总损失: {loss:.3e}, PDE: {pde:.3e}, '
                  f'内边界: {bc_inner:.3e}, 外边界: {bc_outer:.3e}')
        theta_init = torch.cat([p.flatten() for p in self.parameters()]).cpu().detach().numpy()
        result = minimize(fun=self.get_loss_and_grad,
                         x0=theta_init,
                         method='L-BFGS-B',
                         jac=True,
                         callback=callback,
                         options={'maxiter': maxiter,
                                 'maxfun': 50000,
                                 'maxcor': 50,
                                 'maxls': 50,
                                 'ftol': 1.0 * np.finfo(float).eps})
        start = 0
        for param, size in zip(self.parameters(), self.sizes):
            param.data = torch.tensor(result.x[start:start+size], dtype=torch.float32, device=self.device).reshape(param.shape)
            start += size

    def train(self, n_iter):
        self.xavier_init()
        
        adaptor = LR_Adaptor(self.optimizer_adam, loss_weight=self.loss_weights, 
                             num_pde=1, alpha=0.1, mode="max")
        
        start_time = time.time()
        for it in range(n_iter):
            def closure(skip_backward=False):
                if not skip_backward:
                    self.zero_grad()
                    loss, loss_pde, loss_bc_inner, loss_bc_outer = self.compute_loss(self.x, self.z)
                    return loss
                
                u_real_pred, u_imag_pred, f_loss = self.net_helmholtz(self.x, self.z)
                
                loss_pde = torch.mean(f_loss)
                
                r_mapped = torch.sqrt(self.x**2 + self.z**2)
                bc_inner_mask = (torch.abs(r_mapped - radius) < 0.01).float()
                loss_bc_inner = torch.mean(bc_inner_mask * ((u_real_pred - bc_value)**2 + u_imag_pred**2))
                
                bc_outer_mask = (torch.abs(r_mapped - outer_radius) < 0.01).float()
                dudx_real, dudx_imag = complex_gradients(u_real_pred, u_imag_pred, self.x)
                dudz_real, dudz_imag = complex_gradients(u_real_pred, u_imag_pred, self.z)
                dudr_real = (self.x * dudx_real + self.z * dudz_real) / r_mapped
                dudr_imag = (self.x * dudx_imag + self.z * dudz_imag) / r_mapped
                loss_bc_outer = torch.mean(bc_outer_mask * ((dudr_real + k * u_imag_pred)**2 + (dudr_imag - k * u_real_pred)**2))
                
                adaptor.losses = torch.stack([loss_pde, loss_bc_inner, loss_bc_outer])
                
                return None
            
            loss = adaptor.step(closure)
            
            if it % 100 == 0:
                _, loss_pde, loss_bc_inner, loss_bc_outer = self.compute_loss(self.x, self.z)
                misfit.append(loss.item())
                elapsed = time.time() - start_time
                print(f'迭代: {it}, 总损失: {loss.item():.3e}, PDE: {loss_pde.item():.3e}, '
                      f'内边界: {loss_bc_inner.item():.3e}, 外边界: {loss_bc_outer.item():.3e}, '
                      f'权重: {adaptor.loss_weight.tolist()}, 时间: {elapsed:.2f}')
                start_time = time.time()
        
        self.loss_weights = adaptor.loss_weight.clone()
        print(f"Adam 阶段结束，自适应权重: {self.loss_weights.tolist()}")
        
        print("开始 L-BFGS 优化...")
        self.train_lbfgs(maxiter=5000)

    def predict(self, x_star, z_star):
        x_star = torch.tensor(x_star, dtype=torch.float32, requires_grad=True).to(self.device)
        z_star = torch.tensor(z_star, dtype=torch.float32, requires_grad=True).to(self.device)
        u_real, u_imag, _ = self.net_helmholtz(x_star, z_star)
        return u_real.detach().cpu().numpy(), u_imag.detach().cpu().numpy()

def exact_solution(x, z, k=1.0, a=1.0, phi0=100.0):
    r = np.sqrt(x**2 + z**2)
    hankel = hankel1(0, k * r)
    hankel_bc = hankel1(0, k * a)
    phi = phi0 * hankel / hankel_bc
    return np.real(phi), np.imag(phi)

def generate_training_data_lhs_direct():
    """直接在 r ∈ [radius, outer_radius] 区域使用拉丁超立方采样生成训练点"""
    lhs_samples = lhs(2, samples=n_train)
    r = radius + lhs_samples[:,0] * (outer_radius - radius)
    theta = 2.0 * np.pi * lhs_samples[:,1]
    x_train = r * np.cos(theta)
    z_train = r * np.sin(theta)

    n_bc = int(n_train * 0.1)
    theta_bc = np.linspace(0, 2*np.pi, n_bc)
    x_bc = radius * np.cos(theta_bc)
    z_bc = radius * np.sin(theta_bc)

    x_all = np.concatenate([x_train, x_bc])
    z_all = np.concatenate([z_train, z_bc])
    return x_all.reshape(-1,1), z_all.reshape(-1,1)

if __name__ == "__main__":
    x_train, z_train = generate_training_data_lhs_direct()
    layers = [2, 50, 50, 50, 50, 2]
    model = PhysicsInformedNN(x_train, z_train, layers, omega)
    print("开始训练...")
    model.train(n_iter)

    n_test = 200
    r_test = np.linspace(radius, outer_radius, n_test)
    theta_test = np.linspace(0, 2*np.pi, n_test)
    R, Theta = np.meshgrid(r_test, theta_test)
    x_test = R * np.cos(Theta)
    z_test = R * np.sin(Theta)

    u_real_pred, u_imag_pred = model.predict(x_test.flatten().reshape(-1,1), z_test.flatten().reshape(-1,1))
    u_real_pred = u_real_pred.reshape(n_test, n_test)

    u_real_exact, _ = exact_solution(x_test.flatten(), z_test.flatten())
    u_real_exact = u_real_exact.reshape(n_test, n_test)

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.contourf(x_test, z_test, u_real_pred, levels=50, cmap='jet')
    plt.colorbar(label='压力 (Pa)')
    plt.title('PINN 实部')

    plt.subplot(1,3,2)
    plt.contourf(x_test, z_test, u_real_exact, levels=50, cmap='jet')
    plt.colorbar(label='压力 (Pa)')
    plt.title('解析解实部')

    plt.subplot(1,3,3)
    error = np.abs(u_real_pred - u_real_exact)
    plt.contourf(x_test, z_test, error, levels=50, cmap='inferno')
    plt.colorbar(label='误差 (Pa)')
    plt.title(f'最大误差: {error.max():.2f} Pa')

    plt.tight_layout()
    plt.savefig('pinn_results.png', dpi=300)

    plt.figure(figsize=(8,6))
    plt.plot(misfit, label='训练损失')
    plt.yscale('log')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('训练过程中的损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png', dpi=300)