import sys
import petsc4py
# 最稳妥的初始化方式（如果你的脚本里没显式调用，先试试）
petsc4py.init(sys.argv)
from petsc4py import PETSc
PETSc.Sys.Print("PETSc version: %s" % PETSc.Sys.getVersion())
PETSc.Sys.Print("MPI size=%d rank=%d" % (PETSc.COMM_WORLD.getSize(), PETSc.COMM_WORLD.getRank()))