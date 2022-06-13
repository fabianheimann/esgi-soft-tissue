import netgen.geom2d as geom2d
from ngsolve import *
geo = geom2d.SplineGeometry()
pnums = [geo.AddPoint (x,y,maxh=0.01) for x,y in [(0,0), (1,0), (1,0.1), (0,0.1)] ]
for p1,p2,bc in [(0,1,"bot"), (1,2,"right"), (2,3,"top"), (3,0,"left")]:
     geo.Append(["line", pnums[p1], pnums[p2]], bc=bc)
mesh = Mesh(geo.GenerateMesh(maxh=0.05))

E, nu = 210, 0.2
mu  = E / 2 / (1+nu)
lam = E * nu / ((1+nu)*(1-2*nu))

#def C(u):
    #F = Id(2) + Grad(u)
    #return F.trans * F

#def NeoHooke (C):
    #return 0.5*mu*(Trace(C-Id(2)) + 2*mu/lam*Det(C)**(-lam/2/mu)-1)

def sigma(u):
    return lam*Trace(Grad(u))*Id(2) + mu*(Grad(u) + (Grad(u)).trans)

factor = Parameter(0)
factor.Set(0.4)
force = CoefficientFunction( (0,factor) )

fes = H1(mesh, order=4, dirichlet="left", dim=mesh.dim)
u = fes.TrialFunction()
v = fes.TestFunction()

a = BilinearForm(fes, symmetric=True)
a += InnerProduct( sigma(u), Grad(v))*dx

L = LinearForm(fes)
L += force*v*dx

u = GridFunction(fes)
u.vec[:] = 0

a.Assemble()
L.Assemble()

u.vec.data = a.mat.Inverse(fes.FreeDofs() )*L.vec

Draw(u, mesh, "def")
#Draw (CF(1), mesh, "C(u)")
