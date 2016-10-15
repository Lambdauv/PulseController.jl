# Clifford module

# This module contains code for generating sequences of clifford gates.  The
# exported functions benchmark1Qubit and benchmark2Qubit generate sequences
# equivalent to an idle gate for one qubit, and to either an idle or swap for
# two qubits.  Additional arguments allow for restricting the control sequences
# to only XY control and for interleaved random benchmarking of a specific gate.



#============= Matrices =============#

# Mathematical background for using integers mod 5 instead of complexes:

# 2^n x 2^n Clifford matrices can be normalized to have all entries one of
# {0, +1, -1, +i, -i}, and the first nonzero entry in the first row may always
# be 1.  This is a 5-element set closed under multiplication, so we can just use
# the 5-element field. Explicitly: 0 <=> 0, 1 <=> 1, -1 <=> 4, i <=> 2, -i <=> 3
# respects element multiplication.  Matrix multiplication involves addition as
# well, and while 1 + 2 = 3, 1 + i is not -i.  However, under the normalization
# convention above, the entire matrix is divided by the first nonzero entry, or
# equivalently multiplied by its inverse.  Thus, one need only verify that all
# encodings of {1,2,3,4} with magnitudes sqrt(2)^k for integer k have the same
# angular orientation (that is, they read 1, 2, 4, 3 taken clockwise).  This
# may be easily proven graphically, showing that this isomorphism is sound.

# The clifford matrix is 2^2n elements and all are part of the 5-element field,
# so in principle an enitre 4x4 matrix is one UInt64's worth of information.
# In practice, that kind of bit twiddling is likely slower than machine
# vectorized integer operations, so instead we will just use matrices
# of UInt8s, taking things mod 5 at the end of a calculation.

module Clifford
export benchmark1Qubit
export benchmark2Qubit
export Pulse
export gateNames

# Andrew's pull request uses SMatrix and creates CMatrix type instead of alias,
# resulting in Base compatibility and ~5x speedup with computations.
typealias CMatrix Matrix{UInt8}

# Our normalization convention is that the first nonzero entry be a 1.  This
# normalization reconciles the 5-element field and the +1, -1, +i, -i
# representations, showing both are isomorphic to the Clifford group.
# NOTE: normalize only produces correct results on matrices already modded by 5.
function normalize(A::CMatrix)
  normConst = A[findfirst(A)]
  inverse = normConst $ ((normConst >>> 1) & 1) # Inverse mod 5
  mod(A*inverse, 5)
end

# Implement multiplication such that the matrix product is properly normalized.
function mult(A::CMatrix, B::CMatrix...)
  normalize(reduce((B,C) -> mod(B*C, 5), A, B))
end

# We also have to modify matrix inversion.  My present choice is to compute the
# standard inverse and multiply it by the determinant to get a strictly-integral
# matrix.  Then we normalize.  More direct routes could be taken to get the
# minors of each element but I feel it would amount to rewriting too much for
# something not performance-critical.
function invert(A::CMatrix)
  normalize(map(x -> UInt8(round(mod(x, 5))), inv(A) * det(A)))
end

# Output a clifford matrix corresponding to A on left set of qubits and B on
# right set of qubits - used for 2-qubit representation of independent 1-qubit
# gates.
function outer(A::CMatrix, B::CMatrix)
  mod(kron(A, B), 5)
end

# I'll explicitly write out the x pi/2 and y pi/2 single-qubit matrices, using
# these to generate the rest.  I will refer to them as f and g, respectively.
const f = UInt8[1 2; 2 1]
const g = UInt8[1 4; 1 1]
const h = mult(g, f, g, g, g) # Gives the z/2 rotation

# For later use, the 2-qubit matrices cz and swap
const cz = UInt8[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 4]
const swap = UInt8[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]


# Single qubit clifford group is 24 gates, I'll just do this the long way.
# In the Mathematica notebook these are shown to correspond 1:1 with the pulses
# listed below (keepiing in mind that pulses are time-ordered left to right,
# while matrices are applied right to left)
const SQClif = CMatrix[eye(f),
               f,
               g,
               h,
               mult(f, f),
               mult(g, g),
               mult(h, h),
               mult(f, f, f),
               mult(g, g, g),
               mult(h, h, h),
               mult(f, g),
               mult(f, g, g, g),
               mult(f, f, f, g),
               mult(f, f, f, g, g, g),
               mult(f, h),
               mult(f, h, h, h),
               mult(f, f, f, h),
               mult(f, f, f, h, h, h),
               mult(f, f, g),
               mult(f, f, g, g, g),
               mult(g, g, h),
               mult(g, g, h, h, h),
               mult(h, h, f),
               mult(h, h, f, f, f)]


# 2-qubit clifford group can be constructed as a SQClif gate on each qubit,
# followed by one of the ten operations below, followed by an optional swap.
# We use only the "easy" half of the clifford group, modding out by swaps.
const EntanglingGate = CMatrix[eye(cz),
                       cz,
                       mult(outer(eye(f), f), cz),
                       mult(outer(eye(f), g), cz),
                       mult(outer(f, eye(f)), cz),
                       mult(outer(f, f), cz),
                       mult(outer(f, g), cz),
                       mult(outer(g, eye(f)), cz),
                       mult(outer(g, f), cz),
                       mult(outer(g, g), cz)]


#============= Pulses =============#

# We need an understandable way to designate the gates, so that control lines
# can quickly convert between this program's output and pulse sequencing.  A
# pulse matrix has each column a different qubit, and each row a different time
# step.  Functions closer to the instruments will convert the entries for a
# specific qubit into microwave control pulses, while it is the responsibility
# of the higher-up code to ensure that the requested pulses are all timed
# appropriately.

# Initially Pulse was simply a typealias to Matrix{Int8}, but I wanted to treat
# Pulses differently in some special cases for dispatch.  The definitions below
# are sufficient to allow Julia to treat Pulse objects identically to Matrices
# for indexing and concatenation, and show Pulses nicely with named gates.  It
# also allows one to specify a sequence with one pulse repeated many times, for
# example [Xpi; Idle*10; Ypi] for a pause between Xpi and Ypi of 10 Idle gates.
import Base: size, getindex, setindex!, linearindexing, similar, showarray, *
type Pulse <: AbstractMatrix{Int8}
    m::Matrix{Int8}
end
size(x::Pulse) = size(x.m)
getindex(x::Pulse, i::Int) = getindex(x.m, i)
getindex(x::Pulse, i::Int...) = getindex(x.m, i...)
linearindexing(::Type{Pulse}) = Base.LinearFast()
getindex(x::Pulse, I...) = getindex(x.m, I...)
setindex!(x::Pulse, v, i::Int) = x.m[i] = v
setindex!(x::Pulse, v, I::Int...) = x.m[I...] = v
setindex!(x::Pulse, v, I...) = x.m[I...] = v
similar{T}(x::Pulse, ::Type{T}, dims::Dims) = Pulse(Array{Int8}(dims))
*(x::Pulse, I::Integer) = vcat(fill(x, I)...)
*(I::Integer, x::Pulse) = vcat(fill(x, I)...)

# This scheme for getting pulses to display nicely in the REPL is hackish and
# unstable.  Hopefully in later releases it won't be necessary.  TODO: better..
showarray(io::IO, x::Pulse, repr::Bool) = prettyPrint(x)
function prettyPrint(x::Pulse)
  println("Length-"*string(size(x,1))*" pulse sequence:")
  [println(reduce(*,"",[gateNames[x[y,z]]*"\t" for z in 1:size(x,2)]))
                                               for y in 1:size(x,1)]
  nothing
end

# This is essentially an enum, but the numbers are chosen to have some later
# significance.  Note the null gate is empty rather than idle.
const nul = Pulse(Array(Int8,(0,1)))
const Idle =  Pulse(fill(Int8(7),(1,1)))
const Xpi2 =  Pulse(fill(Int8(1),(1,1)))
const Xpi =   Pulse(fill(Int8(5),(1,1)))
const X3pi2 = Pulse(fill(Int8(3),(1,1)))
const Ypi2 =  Pulse(fill(Int8(2),(1,1)))
const Ypi =   Pulse(fill(Int8(6),(1,1)))
const Y3pi2 = Pulse(fill(Int8(4),(1,1)))
const Zpi2 =  Pulse(fill(Int8(8),(1,1)))
const Zpi =   Pulse(fill(Int8(9),(1,1)))
const Z3pi2 = Pulse(fill(Int8(10),(1,1)))
const CZ =    Pulse(Int8[11 11])  # CZ requires 2 qubits to implement.

# While the code works with numbers, it is handy to see the gates as strings.
const gateNames = ["Xpi2", "Ypi2", "X3pi2", "Y3pi2", "Xpi", "Ypi", "Idle",
"Zpi2", "Zpi", "Z3pi2", "CZ"]


# There are two potentially desirable ways to perform clifford gates.  One is
# in the fewest number of gates, assuming XY and Z lines are present (WithZ).
# The other is in the fewest number of XY-only gates (NoZ).  In the WithZ case,
# the program will be able to do RB seqeunces with the same effect in fewer
# gates, and the Qubit type will ask for the channel for all 3 lines: XYI, XYQ
# and Z.  In the NoZ case, the gates performed will only exist on the
# XYI and XYQ lines, and the Qubit type will not ask about a Z line.  The latter
# is necessary for doing single qubit RB with only the AWG, since there are 4
# lines and 2 are used for readout.
const SQPulseWithZ = Pulse[nul,
      Xpi2,
      Ypi2,
      Zpi2,
      Xpi,
      Ypi,
      Zpi,
      X3pi2,
      Y3pi2,
      Z3pi2,
      [Ypi2; Xpi2],
      [Y3pi2; Xpi2],
      [Ypi2; X3pi2],
      [Y3pi2; X3pi2],
      [Zpi2; Xpi2],
      [Z3pi2; Xpi2],
      [Zpi2; X3pi2],
      [Z3pi2; X3pi2],
      [Ypi2; Xpi],
      [Y3pi2; Xpi],
      [Zpi2; Ypi],
      [Z3pi2; Ypi],
      [Xpi2; Zpi],
      [X3pi2; Zpi]]

const SQPulseNoZ = Pulse[nul,
      Xpi2,
      Ypi2,
      [Xpi2; Ypi2; X3pi2],
      Xpi,
      Ypi,
      [Xpi; Ypi],
      X3pi2,
      Y3pi2,
      [Xpi2; Y3pi2; X3pi2],
      [Ypi2; Xpi2],
      [Y3pi2; Xpi2],
      [Ypi2; X3pi2],
      [Y3pi2; X3pi2],
      [Xpi2; Ypi2],
      [Xpi2; Y3pi2],
      [X3pi2; Y3pi2],
      [X3pi2; Ypi2],
      [Ypi2; Xpi],
      [Y3pi2; Xpi],
      [Y3pi2; Xpi2; Y3pi2],
      [Ypi2; Xpi2; Ypi2],
      [Ypi; Xpi2],
      [Ypi; X3pi2]]

# With the above defined, we can make a lookup dictionary for the recovery pulse
SQLookupWithZ = Dict{CMatrix, Pulse}(map(=>, SQClif, SQPulseWithZ))
SQLookupNoZ = Dict{CMatrix, Pulse}(map(=>, SQClif, SQPulseNoZ))

# And with only slightly more effort, we can do the same for the 2-qubit case.
# We need to list the pulses for the entangling gates:
const EntanglingPulse = Pulse[[nul nul],
                   CZ,
                   [CZ; [Idle Xpi2]],
                   [CZ; [Idle Ypi2]],
                   [CZ; [Xpi2 Idle]],
                   [CZ; [Xpi2 Xpi2]],
                   [CZ; [Xpi2 Ypi2]],
                   [CZ; [Ypi2 Idle]],
                   [CZ; [Ypi2 Xpi2]],
                   [CZ; [Ypi2 Ypi2]]]


# Coordinate pulses of unknown lengths on two qubits in parallel.  This is in
# complement to "outer" defined above, in that if pulse A has matrix mA, B has
# matrix mB, then pulseOuter(A, B) has matrix given by outer(mA, mB).
function pulseOuter(A::Pulse, B::Pulse)
  if A == nul
    if B != nul
      A = Idle
    else
      return [A B]
    end
  elseif B == nul
    B = Idle
  end

  pad = size(A, 1) - size(B, 1)
  if pad == 2
    [A [B; Idle; Idle]]
  elseif pad == 1
    [A [B; Idle]]
  elseif pad == -1
    [[A; Idle] B]
  elseif pad == -2
    [[A; Idle; Idle] B]
  else
    [A B]
  end
end

# And then the analogous (5760-element) lists and lookup table.  Note that we
# only employ the WithZ case here, assuming qubits that are coupled need to be
# Z-tunable.  With a gmon scheme this might not be accurate.
const TQClif = collect([mult(EntanglingGate[k], outer(SQClif[i], SQClif[j]))
          for i=1:24, j=1:24, k=1:10])
const TQPulse= collect([[pulseOuter(SQPulseWithZ[i], SQPulseWithZ[j]); EntanglingPulse[k]]
          for i=1:24, j=1:24, k=1:10])

# The lookup table finds the pulse corresponding to either the matrix given or
# that matrix plus a swap operation.  Since we use this lookup table to reach
# the ground state, both will work.  We will not be turning error states into
# clean ones, just switching which qubit fails to read ground state.
TQLookup = Dict{CMatrix, Pulse}(map(=>, [TQClif map(A->mult(swap, A), TQClif)]
                                      , [TQPulse TQPulse]))


# From this we can build a function which chooses uniformly from the group,
# building sequences of nClifs operations ultimately ending in the identity.
# While we treat the 1-qubit and 2-qubit cases separately, it is mostly
# copy-paste.

# These functions take a gate to benchmark, given by its index in the SQClif or
# TQClif lists.  These functions are not meant to be called directly by users,
# who would not know this index.  Calling functions should find the index using
# find() or findfirst() and some more meaningful description of the gate, either
# the matrix in *Clif or the pulse sequence in *Pulse.  For a baseline, the null
# gate should be used, which has empty lists as pulses.  This is index 1.

# This leads to a bit of complication if we want to benchmark the idle gate,
# which explicitly idles for a short time instead of proceeding directly to the
# next clifford.  One possibility is to add to the gateset an Idle pulse which
# has the same matrix but a different pulse and different index.  It would be
# left out of the lookup dictionary.  TODO: add support for benchmarking "idle."

function benchmark1Qubit(nClifs, pulseIndex::Int = 1, ZControl::Bool = true)
  selection = rand(1:24, nClifs-1)
  recovery = invert(mult([SQClif[reverse(selection)]
                          fill(SQClif[pulseIndex], nClifs-1)]'[:]...))
  if ZControl
    [vcat(SQPulseWithZ[[selection fill(pulseIndex, nClifs-1)]'[:]]...); SQLookupWithZ[recovery]]
  else
    [vcat(SQPulseNoZ[[selection fill(pulseIndex, nClifs-1)]'[:]]...); SQLookupNoZ[recovery]]
  end
end
# Convenience method allowing ommitted index but specified ZControl status
function benchmark1Qubit(nclifs, ZControl::Bool, pulseIndex::Int = 1)
  benchmark1Qubit(nclifs, pulseIndex, ZControl)
end

# 2 qubit gates cannot be done without Z control unless gmon-coupled, and this
# code doesn't presently support sending pulses to interqubit couplings.  As
# such, ZControl is assumed true always.
function benchmark2Qubit(nClifs, pulseIndex::Int = 1)
  selection = rand(1:5760, nClifs-1)
  recovery = invert(mult([TQClif[reverse(selection)]
                          fill(TQClif[pulseIndex], nClifs-1)]'[:]...))
  [vcat(TQPulse[[selection fill(pulseIndex, nClifs-1)]'[:]]...); TQLookup[recovery]]
end

end # End module Clifford