import torch

class Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
        w = x * y * z
        out = x * y + y * z + w
        ctx.save_for_backward(x, y, out)
        ctx.z = z  # z is not a tensor
        ctx.w = w  # w is neither input nor output
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, y, out = ctx.saved_tensors
        z = ctx.z
        gx = grad_out * (y + y * z)
        gy = grad_out * (x + z + x * z)
        gz = None
        return gx, gy, gz
    
    @staticmethod
    def forward_without_backward(ctx):
        x, y, out = ctx.saved_tensors
        z = ctx.z
        return x, y, out, z
    

a = torch.tensor(1., requires_grad=True, dtype=torch.double)
b = torch.tensor(2., requires_grad=True, dtype=torch.double)
c = 4
d = Func.apply(a, b, c)
print(d)
# x, y, z = d.backward()
d.backward()
print(a.grad, b.grad)
# print(x, y, z)