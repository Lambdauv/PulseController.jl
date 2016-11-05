#========================== The Waveform DataTypes ============================#
# There are two approaches we could take for storing waveforms.

# 1) A fast approach with assumptions.
#       In this, each named gate pulse just has one associated real waveform.
#       If it is an X or Y gate, the waveform is the window on a specific-
#       frequency IQ wave, and the Z is off during that pulse.  If it is a Z
#       gate, then the waveform is the signal sent to the Z line, and the XY
#       line is off during that pulse.  Due to the fact that XY signals must
#       be mutiplied with the IQ wave before rounding and sending to the DAC,
#       it is only sensible to work with real-valued waveforms in this approach.
#       In this case, all pulses must be the same duration, so that sequencing
#       code can splice in a generic idle pulse for the inactive lines.

# 2) A full approach with no assumptions but more data to optimize
#       In this, each named gate pulse has 3 associated integer waveforms,
#       designating the XYI, XYQ, and Z pulse shapes.  This allows for tactics
#       such as DRAG.  It is also "closer to the machine," in that the data
#       being held is exactly the digital data sent to the DAC, rather than
#       needing to be altered in some way beforehand.  For sequencing, all
#       lines draw from the waveform data for the gate at hand, meaning all
#       gates can nominally be independent lengths.

# In the event we are controlling a qubit with XY only, we need to adapt the
# ExactWaveform type to hold XYI and XYQ but not Z.  This will consist of length
# matched pulses on XYI and XYQ and an empty pulse on Z.  The lengths do not
# need to match the idle length, because the Z line simply doesn't exist.  Due
# to this necessity, I have removed the strict length-matching conditions on
# newly created ExactWaveform objects.  TODO: Find consistent set of checks
# for errors in initialization.  I don't want users sending pulses out of phase.

# When doing PSO and other optimization, it seems sensible to begin with the
# first approach to get a good gate, and then tune it up further with the second
# to get a by-point optimized gate.  Using Julia typing, we can make the switch
# painless.

# Pulses (equivalently, gates) live independently of which qubits they are on,
# so it seems the responsibility of the qubit type to maintain a dictionary of
# pulse => waveform, with waveform one of the two above collections of data.
# We can maintain the pulse typealias from above, which plays very nicely with
# sequencing (a sequence of multiple pulses is the same type as a single pulse),
# and this potentially gives us the ability to add to the qubit's dictionary any
# optimized waveform for arbitrary pulse sequences.
module Waveforms

export Waveform
export FloatWaveform
export ExactWaveform
export floatIdleLength

const floatIdleLength = 30

abstract Waveform

type FloatWaveform <: Waveform
  wavedata::Vector{Float64}
end

type ExactWaveform <: Waveform
  XYI::Vector{UInt16}
  XYQ::Vector{UInt16}
  Z::Vector{UInt16}
  dirty::Bool # Denotes whether this is up to date with the DAC's memory

  # Provide an inner constructor.  There are two options for an ExactWaveform.
  # If it encodes a FloatWaveform's data, it will have empty pulses on the
  # inactive lines.  All active lines must be length matched to the idle pulse.
  # If not, the only constraint is that all lines must be length matched to
  # one another.  Hence, 3 lines of comparators.  Julia allows x == y == z.

  # 9/9/16 - disabled constructor to accomodate exact XY-only control.
#=  ExactWaveform(XYI, XYQ, Z, dirty) =
    (length(XYI) == length(XYQ) == length(Z) ||
     length(XYI) == length(XYQ) == floatIdleLength && length(Z) == 0 ||
     length(XYI) == length(XYQ) == 0 && length(Z) == floatIdleLength) ?
    new(XYI, XYQ, Z, dirty) :
    error("Incompatible waveform lengths") =#
end

# Make ExactWaveforms readable
import Base.string
function string(x::ExactWaveform)
  str = x.dirty ? "Unsynced" : "Synced"
  str *= " waveform with the following pulse shapes:\nXYI: "
  str *= length(x.XYI) > 0 ? string([Int(c) for c in x.XYI]) : "idle"
  str *= "\nXYQ: "
  str *= length(x.XYQ) > 0 ? string([Int(c) for c in x.XYQ]) : "idle"
  str *= "\nZ: "
  str *= length(x.Z) > 0 ? string([Int(c) for c in x.Z]) : "idle/undefined"
end

import Base.print
print(io::IO, x::ExactWaveform) = print(io, string(x))

import Base.show
show(io::IO, x::ExactWaveform) = print(io, x)

end # End module Waveforms