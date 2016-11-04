module PulseController

# Bring in other files as submodules.  Every file is exactly one module.
# The dependency tree is as follows:

# Clifford - standalone
# Waveforms - standalone

# Qubits - imports from Clifford and Waveforms

# Seqeuncing - imports from Qubits

# ParticleSwarm - standalone

# Orbit - imports from Sequencing and ParticleSwarm

include("clifford.jl")
include("waveforms.jl")
include("qubits.jl")
include("sequencing.jl")
include("particleswarm.jl")
include("orbit.jl")

using .ORBIT # Carries all other modules

export sendSequence

# For writing out sequences, we will provide all of the pulse names.  Hopefully
# having these in scope doesn't clash with user variables
import .Clifford: Idle, Xpi, Xpi2, X3pi2, Ypi, Ypi2, Y3pi2, Zpi, Zpi2, Z3pi2, CZ
export Idle, Xpi, Xpi2, X3pi2, Ypi, Ypi2, Y3pi2, Zpi, Zpi2, Z3pi2, CZ

# From qubits
export Qubit
export cosInit
export gaussInit
export generalInit

# From waveforms, for setting qubit waveforms manually
export FloatWaveform
export ExactWaveform

end # End module PulseController

using PulseController # Bad style probably