from netgen.csg import *
from ngsolve import *
from xfem import *
from xfem.lsetcurv import *

mesh = Mesh("meniscous.vol.gz")
print(mesh.GetBoundaries())
lsetmeshadap1 = LevelSetMeshAdaptation(mesh, order=2, threshold=0.1,
                                    discontinuous_qn=True)
lsetmeshadap2 = LevelSetMeshAdaptation(mesh, order=2, threshold=0.1,
                                    discontinuous_qn=True)

bottom_find_lset = z - 6.456
lsetmeshadap1.CalcDeformation(bottom_find_lset)
bottom_find_ci = CutInfo(mesh, lsetmeshadap1.lset_p1)
#Draw(lsetmeshadap.lset_p1, mesh, "bottom_find_lset")
bot = bottom_find_ci.GetElementsOfType(HASNEG)
#exit()
#Draw(bottom_find_lset, mesh, "bottom_find_lset")

levelset = x + 10 - 0.3*(y +11)# + 0.3*(x-0.5)**2 + 0.2*sin(pi*(x-0.4))
deformation = lsetmeshadap2.CalcDeformation(levelset)
lsetp1 = lsetmeshadap2.lset_p1
Draw(lsetp1, mesh, "lsetp1")

ci = CutInfo(mesh, lsetp1)
hasneg = ci.GetElementsOfType(HASNEG)
haspos = ci.GetElementsOfType(HASPOS)

E, nu = 210, 0.2
mu  = E / 2 / (1+nu)
lam = E * nu / ((1+nu)*(1-2*nu))

def eps(u):
    return (Grad(u) + (Grad(u)).trans)/2

def sigma(u):
    return lam*Trace(Grad(u))*Id(3) + 2*mu*eps(u)

factor = Parameter(0)
factor.Set(-0.4)
force = CoefficientFunction( (0,0,factor) )

fes = H1(mesh, order=2, dim=mesh.dim, dgjumps=True)
#fes_neg = H1(mesh, order=2, dirichlet="left", dim=mesh.dim)
#fes_neg = Compress(fes, GetDofsOfElements(fes,hasneg))
#fes_pos = Compress(fes, GetDofsOfElements(fes,haspos))
#VhG = fes_neg * fes_pos
VhG = fes*fes

fd = BitArray(2*fes.ndof)
fd[0:fes.ndof] = GetDofsOfElements(fes,hasneg) & ~GetDofsOfElements(fes,bot)
fd[fes.ndof:2*fes.ndof] = GetDofsOfElements(fes,haspos) & ~GetDofsOfElements(fes,bot)

u, v = VhG.TnT()
n = 1.0 / grad(lsetp1).Norm() * grad(lsetp1)
kappa = (CutRatioGF(ci), 1.0 - CutRatioGF(ci))
h = specialcf.mesh_size

dx1 = dCut(lsetp1, NEG)
dx2 = dCut(lsetp1, POS)
ds = dCut(lsetp1, IF)
dw = dFacetPatch(definedonelements=GetFacetsWithNeighborTypes(mesh, a=hasneg, b=ci.GetElementsOfType(IF)))

average_flux_u = sum([kappa[i]*sigma(u[i]) * n for i in [0, 1]])
average_flux_v = sum([kappa[i]*sigma(v[i]) * n for i in [0, 1]])

depth = Parameter(1.)

a = BilinearForm(VhG, symmetric=True)
a += InnerProduct( sigma(u[0]), eps(v[0]) )*dx1
a += InnerProduct( sigma(u[1]), eps(v[1]) )*dx2
a += (1 - IfPos(y-depth,0,1)*IfPos(z-10.5,1,0))*(-InnerProduct( average_flux_u, (v[0] - v[1])) - InnerProduct( average_flux_v, (u[0]-u[1])) + 50./h*E*(v[0]-v[1])*(u[0]-u[1]))*ds
a += 0.5 / h**2 * (u[0] - u[0].Other()) * (v[0] - v[0].Other()) * dw
a += 0.5 / h**2 * (u[1] - u[1].Other()) * (v[1] - v[1].Other()) * dw

L = LinearForm(VhG)
L += force*v[0]*dx1
L += force*v[1]*dx2

u = GridFunction(VhG)

vtk = VTKOutput(ma=mesh, coefs=[lsetp1, u.components[0] , u.components[1], sigma(u.components[0]), sigma(u.components[1]) ], names = ["lset","u1","u2", "sigma1", "sigma2"], filename="result_men", subdivision=1)

Draw(x, mesh, "x")
Draw(y, mesh, "y")
Draw(z, mesh, "z")

for i in range(0, 350):
    depth.Set(-14.5 + i/40)
    a.Assemble()
    L.Assemble()

    u.vec.data = a.mat.Inverse(fd)*L.vec
    #u.vec.data = a.mat.Inverse(VhG.FreeDofs())*L.vec
    vtk.Do()
    print("Done with i = ", i)
