module Sequencing

using ..Qubits # Includes Clifford and Waveforms
using InstrumentControl
using InstrumentControl.AWG5014C
using Alazar # For readout

export sendSequence

# From qubits
export Qubit
export cosInit
export gaussInit
export generalInit

# From waveforms
export FloatWaveform
export ExactWaveform

# From clifford
export benchmark1Qubit
export benchmark2Qubit
export Pulse

# gateNames has its last use here in naming the AWG pulse - no parents need it
 
#============================ Basic seqeuncing ================================#
# The sequencing also has to be different depending on which type
# of waveform we are using.  In the case of FloatWaveforms, we need to insert
# idle pulses regularly, which are all the same point in the DAC's memory.
# In the case of ExactWaveforms, we need to access distinct points in the DAC's
# memory for each combination of (line, pulse).  The code allows
# for the possibility that the qubit's lines are not all on the same board.

# In preparing for 1-qubit sequencing, the qubit ensures that all pulses in its
# dictionary are up to date, and updates the 10x3 matrix of indices.
# These map the 10 basic single qubit pulses to their corresponding on-board
# waveform index.  That means that by calling q.pulseConvert[seq, n], with some
# sequence generated elsewhere in the code, we get the actual index list for 
# sequencing on the nth line.
function prepForSeq(q::Qubit)
  # Communicate with DAC only if necessary
  for p in q.waveforms
    if p[2].dirty
      q.pulseConvert[p[1],1] = pushWaveform(p[2].XYI, q.lineXYI[1], gateNames[p[1]]*"I")
      q.pulseConvert[p[1],2] = pushWaveform(p[2].XYQ, q.lineXYQ[1], gateNames[p[1]]*"Q")
      if isa(q, QubitWithZ)
        q.pulseConvert[p[1],3] = pushWaveform(p[2].Z, q.lineZ[1])
      end
      p[2].dirty = false
    end
  end
end

# Return the index of the waveform in memory
function pushWaveform(wavedata::Vector{UInt16}, board::Instrument, name="")
  if length(wavedata) == 0
    # This is one of the off channels of a FloatWaveform.  Just return the
    # stored index of idle pulse
    return idles[board]
  end
  if isa(board, InsAWG5014C)
    return awgPush(wavedata, board, name)
  else
    error("Instrument not supported yet, whine at a programmer")
  end
end

# We will be using the VISA protocol and binary writing.  We will also force
# every waveform to 250 points, with front-padded zeros.  A 30-point waveform
# will then have 220 points of offsetValue and then 30 points with signal.
function awgPush(wavedata::Vector{UInt16}, ins::InsAWG5014C, name::String)
  if length(wavedata) > 250
    error("Waveforms longer than 250 points not presently supported.")
  else
    binblockwrite(ins, "WLIST:WAV:DATA "*name*", "*reinterpret(UInt8,
      [fill(offsetValue, 250 - length(wavedata)); map(htol, wavedata)]))
  end
  name # Return the name, how the AWG will refer to it later.
end

# The following functions take pulse sequences in the form produced by the code,
# and send them to the DACs based on their indices in the DAC memory.  This
# means that sendSequence(q, [Xpi2, I, Xpi2]) for example, will send XYI, XYQ,
# and Z pulses according to the prior definitions of these pulses.  
function sendSequence(q::Qubit, sequence::Pulse, readout::Bool=True)
  if size(sequence, 2) == 1
    sendSequence(q, sequence[:], readout)
  else
    error("Please specify only one sequence\n")
  end
end


function sendSequence(q::Qubit, sequence::Vector{Int8}, readout::Bool=True)
  # Make sure we aren't going to send obsolete pulses
  # (no writes done if everything is current)
  prepForSeq(q)

  # TODO: More generalized version of this function.  For now, we assume the
  # qubit is controlled by the AWG, that it is a QubitNoZ object, and that
  # the other AWG lines need to idle until the sequence terminates, and then
  # perform a readout pulse.  This allows the sequencing to be done in one
  # (albeit large) command to the AWG.  If this isn't the goal, bug Brett.
  # Someday readout will be a single pulse for multiple qubits.
  len = "SEQ:LENG "*string(length(sequence))*"\n"
  istring = map((x,y)-> "SEQ:ELEM"*string(x)*":WAV"*string(q.lineXYI[2])*
                                                    " \""*string(y)*"\"\n",
                                1:length(sequence), q.pulseConvert[sequence, 1])
  qstring = map((x,y)-> "SEQ:ELEM"*string(x)*":WAV"*string(q.lineXYQ[2])*
                                                    " \""*string(y)*"\"\n",
                                1:length(sequence), q.pulseConvert[sequence, 2])
  # Get the other channels, which are IQ for readout, and instruct them to be
  # idle at this time
  a = 1:4
  a = a[(a.!=q.lineXYI[2])&(a.!=q.lineXYQ[2])] # Select channels not used by XY
  idle1 = map(x -> "SEQ:ELEM"*string(x)*":WAV"*string(a[1])*" \"I\"\n")
  idle2 = map(x -> "SEQ:ELEM"*string(x)*":WAV"*string(a[1])*" \"I\"\n")
  # Put it all together and send to the AWG
  println(*(len, istring..., qstring..., idle1..., idle2...))
end
#=
function sendSequence(q::Qubit, sequence::Vector{Int8})
  # Make sure we aren't going to send obsolete pulses
  # (no writes done if everything is current)
  prepForSeq(q)
  # Call a lower level function for each line, using indices on the board
  # that correspond to the sequence requested
  seqLowLevel(q.pulseConvert[sequence, 1], q.lineXYI)
  seqLowLevel(q.pulseConvert[sequence, 2], q.lineXYQ)
  if isa(q, QubitWithZ)
    seqLowLevel(q.pulseConvert[sequence, 3], q.lineZ)
  end
  # Turn on output where along this path?

  # How to manage readout?  Do we send a trigger pulse along with the last
  # pulse in the seqeunce, or do we calculate the length of the pulse sequence
  # and just readout based on timing?  Also i expect not all sequences to be
  # sent are necessarily to be followed by readout, so I guess readout-
  # terminated sequences get their own method.
end
=#
# Nothing here covers readout or two-qubit gates.  Both are heirarchically
# lateral from the qubits themselves - pairs of coupled qubits need to know
# waveforms for 2-qubit gates, but not the qubits themselves.  The readout
# process covers all qubits connected to the readout line.  Whatever controls
# the readout will need to be able to tell which qubit is which, but the qubits
# themselves may remain unaware of the readout process.

end # End module sequencing