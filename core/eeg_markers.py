import numpy as np

def export_markers(blocks, vmrk_file, out_file, sampling_rate=5000):
    markers = np.loadtxt(
        vmrk_file,
        comments=';', skiprows=7, delimiter=',',
        dtype=np.str,usecols=range(5))
    v1 = np.where(markers[:,1]=='V  1')[0][0]
    v1_time = int(markers[v1][2])
    last_mk_id = int(markers[-1,0][2:].split('=')[0])
    
    mkid = last_mk_id

    f = open(out_file,'a')
    for block_idx, block in enumerate(blocks):
        mkid += 1
        f.write('Mk%d=Block,Block %d %s,%d,1,0\n'%(mkid, block_idx, block[0], int(v1_time+5000*block[2])))
        mkid += 1
        f.write('Mk%d=Instruction,Instr %d %s,%d,1,0\n'%(mkid, block_idx, block[0], int(v1_time+5000*block[2])))
        mkid += 1
        f.write('Mk%d=Go,Go %d %s,%d,1,0\n'%(mkid, block_idx, block[0], int(v1_time+5000*block[3])))
        for seq in block[-1]:
            mkid +=1
            f.write('Mk%d=Sequence,Seq %s,%d,1,0\n'%(mkid, block[0], int(v1_time+5000*seq[0]['time'])))
            for k in seq:
                mkid +=1
                f.write('Mk%d=Keypress,Key %d,%d,1,0\n'%(mkid, k['key'], int(v1_time+5000*k['time'])))
                if not k['match']:
                    mkid +=1
                    f.write('Mk%d=Error,Error,%d,1,0\n'%(mkid, int(v1_time+5000*k['time'])))
        mkid +=1
        f.write('Mk%d=Rest,Rest %d,%d,1,0\n'%(mkid, block_idx, int(v1_time+5000*block[3])))

    # reexport single volume trigger
    volume_triggers = np.asarray([int(m[2]) for m in markers if (len(m[1])>0 and m[1][0]=='V')])
    for v in volume_triggers:
        mkid += 1
        f.write('Mk%d=VolumeTrigger,VolumeTrigger,%d,1,0\n'%(mkid, v))

    f.close()
