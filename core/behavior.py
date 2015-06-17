import numpy as np
import scipy.io

def load_behavior(in_file, seq_info=None, seq_idx=None, remapping=None):
    
    log = scipy.io.loadmat(in_file)    
    if 'seq_info' in log and seq_info is None:
        seq_info = [(str(seq['seq_desc'][0,0][0]), seq['seq'][0,0][0]) for seq  in log['seq_info'][0]]
    if 'seq_matrx' in log:
        seq_idx = log['seq_matrx'][0]-1

    if seq_idx is None:
        seq_idx = [0]
    
    events = dict()
    keys = np.array([(float(ee[0,0][0]),int(ee[0,2][0]), False, np.inf, np.inf) for ee in log['logoriginal'][0] if ee[0,1]=='rep'],
                    dtype=np.dtype([('time',np.float),('key',np.int),('match',np.bool8),('rt_pre',np.float),('rt_post',np.float)]))
    for ki,k in enumerate(keys):
        if ki>0:
            k[3] = k[0]-keys[ki-1][0]
        if ki<len(keys)-1:
            k[4] = keys[ki+1][0]-k[0]

    if remapping is not None:
        keys['key']=remapping[keys['key']-1]
#    keys_used = np.zeros(len(keys),dtype=np.int)
    blocks = []
    block_idx = 0
    t_instr = -1
    t_go = -1
    for ie,ee in enumerate(log['logoriginal'][0,1:]):
        if len(ee[0])>1:
            log_time = float(ee[0,0][0])
            evt_type = str(ee[0,1][0])
            if evt_type=='START':
                ttl_start = log_time
                keys['time'] -= ttl_start # set key presses to scanner time
            scan_time = log_time - ttl_start
            if evt_type=='Instruction':
                t_instr = scan_time
            elif evt_type=='Practice':
                t_go = scan_time
            elif evt_type=='Rest' and t_go>0:
                t_stop = scan_time
                block_keys_mask = np.logical_and(keys['time']>t_go,keys['time']<t_stop)
                # aggregate keys pressed after the stop signal
                while True:
                    block_keys_idx = np.where(block_keys_mask)[0]
                    last_key = block_keys_idx[-1]
                    first_key = block_keys_idx[0]
                    
                    if last_key<len(keys)-1 and keys[last_key]['rt_post'] < 1.5*keys[block_keys_mask]['rt_post'][:-1].max():
                        block_keys_mask[last_key+1] = True
                    elif first_key>0 and keys[first_key]['rt_pre'] < 1.5*keys[block_keys_mask]['rt_pre'][1:].max():
                        block_keys_mask[first_key-1] = True
                    else:
                        break
#                keys_used[block_keys_mask] += 1
                
                block_keys = keys[block_keys_mask]
                nkey_presses = len(block_keys)
                seq_i = seq_idx[block_idx]
                seq = seq_info[seq_i][1]
                seq_len = seq_info[seq_i][1].size
                seq_name = seq_info[seq_i][0]

                block_seqs = []
                i=0
                while i < len(block_keys):
                    match = np.equal(block_keys[i:i+seq_len]['key'], seq[:nkey_presses-i])
                    if np.all(match):
                        block_keys[i:i+seq_len]['match'] = match
                        block_seqs.append(block_keys[i:i+seq_len])
                        i += seq_len
                    else:
                        i2 = i
                        nomatch = True
                        while nomatch and i < nkey_presses-seq_len:
                            i+=1
                            nomatch = not(np.allclose(block_keys[i:i+seq_len]['key'],seq[:nkey_presses-i]))
                        match2 = np.hstack([np.logical_not(np.cumsum(np.logical_not(match[:i-i2])>0)),
                                            [False]*(i-i2-seq_len)])
                        block_keys[i2:i]['match'] = match2
                        # the added colums shows where it failed in the sequence
                        if i2!=i:
                            block_seqs.append(block_keys[i2:i])
                        elif i >= nkey_presses-seq_len:
                            block_seqs.append(block_keys[i2:])
                            break
                
                blocks.append([seq_name, seq, t_instr, t_go, t_stop,
                               block_keys[0]['time'], block_keys[-1]['time'],
                               block_seqs])
                block_idx += 1            
                t_instr = -1
    return blocks


def behavior_stats(blocks):
    
    stats = dict()
    uniq_seq = np.unique([b[0] for b in blocks])
    
    stats['sequence_rts'] = dict([(seq,[np.diff(np.hstack([sq['time'] for sq in b[-1]])) for b in blocks if b[0]==seq]) for seq in uniq_seq])
