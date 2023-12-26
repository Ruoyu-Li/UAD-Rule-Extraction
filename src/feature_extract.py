from scapy.all import *
import pandas as pd
import numpy as np
import sys
import copy
from datetime import datetime, timedelta
from global_var import *


attack_info = {}
D = ''

buff_dict = {}
X = []

class FlowBuff():
    def __init__(self, tp):
        # tp: (src-ip, dst-ip, sport, dport, proto)
        self.tp = tp
        self.feat_num = 30
        self.active_timeout = 60
        self.inactive_timeout = 15
        # spatial
        self.ps_buff = []
        self.dr_buff = []
        # temporal
        self.ts_init = 0
        self.ts_prev = 0
        self.ts_prev_fwd = 0
        self.ts_prev_bwd = 0
        self.iat_buff = []
        self.iat_buff_fwd = []
        self.iat_buff_bwd = []
        # categorical
        self.proto = tp[4]
        self.dport = tp[2] if tp[2] < tp[3] else tp[3]
        # label
        self.cat = 'N'
        self.sub_cat = 'N'
        self.label = 0

    def get_count(self):
        return len(self.ps_buff)

    def get_fv(self):
        fv = []
        dr_buff = np.array(self.dr_buff)

        # spatial
        # packet count
        fv.append(self.get_count())
        # forward count
        fv.append((dr_buff == 0).sum())
        # backward count
        fv.append((dr_buff == 1).sum())
        # packet size (mean, max, min, var)
        fv += [np.mean(self.ps_buff), np.max(self.ps_buff), 
               np.min(self.ps_buff), np.var(self.ps_buff)]
        # forward size (mean, max, min, var)
        ps_buff_fwd = np.array(self.ps_buff)[dr_buff == 0]
        if len(ps_buff_fwd) == 0:
            fv += [0] * 4
        else:
            fv += [np.mean(ps_buff_fwd), np.max(ps_buff_fwd), 
               np.min(ps_buff_fwd), np.var(ps_buff_fwd)]
        # backward size (mean, max, min, var)
        ps_buff_bwd = np.array(self.ps_buff)[dr_buff == 1]
        if len(ps_buff_bwd) == 0:
            fv += [0] * 4
        else:
            fv += [np.mean(ps_buff_bwd), np.max(ps_buff_bwd), 
               np.min(ps_buff_bwd), np.var(ps_buff_bwd)]
        
        # temporal
        # inter-arrival time (mean, max, min, var)
        if len(self.iat_buff) == 0:
            fv += [0] * 4
        else:
            fv += [np.mean(self.iat_buff), np.max(self.iat_buff), 
                np.min(self.iat_buff), np.var(self.iat_buff)]
        # forward inter-arrival time (mean, max, min, var)
        if len(self.iat_buff_fwd) == 0:
            fv += [0] * 4
        else:
            fv += [np.mean(self.iat_buff_fwd), np.max(self.iat_buff_fwd), 
               np.min(self.iat_buff_fwd), np.var(self.iat_buff_fwd)]
        # backward inter-arrival time (mean, max, min, var)
        if len(self.iat_buff_bwd) == 0:
            fv += [0] * 4
        else:
            fv += [np.mean(self.iat_buff_bwd), np.max(self.iat_buff_bwd), 
                np.min(self.iat_buff_bwd), np.var(self.iat_buff_bwd)]       
        # duration
        fv.append(self.ts_prev - self.ts_init)

        # categorical
        fv.append(self.proto)
        fv.append(self.dport)

        # label
        fv.append(self.cat)
        fv.append(self.sub_cat)
        fv.append(self.label)

        return fv
    
    def clear(self):
        self.ps_buff = []
        self.dr_buff = []
        self.ts_init = 0
        self.ts_prev = 0
        self.ts_prev_fwd = 0
        self.ts_prev_bwd = 0
        self.iat_buff = []
        self.iat_buff_fwd = []
        self.iat_buff_bwd = []
        self.cat = 'N'
        self.sub_cat = 'N'
        self.label = 0

    def set_label_cicids(self, ts):
        """
        return value:
        -1: 不是来自攻击者ip的包，不更新self.label，继续对包进行特征提取
        0: 是来自攻击者ip的包但没有找到记录的攻击时间，不更新self.label，不对包进行特征提取
        1: 是来自攻击者ip的包并且找到了记录的攻击时间，更新self.label，继续对包进行特征提取
        """
        if self.tp[0] in attack_info or self.tp[1] in attack_info:
            try:
                info_list = attack_info[self.tp[0]]
            except:
                info_list = attack_info[self.tp[1]]
            flag = 0
            for item in info_list:
                ts_start, ts_end, cat, sub_cat = item
                if ts_start <= ts <= ts_end:
                    self.cat = cat
                    self.sub_cat = sub_cat
                    self.label = 1
                    flag = 1
                    break
            return flag
        else:
            return -1    

    def set_label_toniot(self, ts):
        """
        TODO
        """
        pass

    def update_and_yield(self, ts, ps, dr):
        """
        处理每个到来的包，增量更新特征提取的队列
        当发生timeout时会返回提取的特征向量，并使用clear()清空所有特征队列
        否则返回空的list
        """
        rtn_fv = []

        if D == 'cicids':
            if self.set_label_cicids(ts) == 0:
                return rtn_fv
        elif D == 'toniot':
            """
            TODO
            """
            pass
        else:
            pass

        if self.ts_init == 0:
            self.ts_init = ts
        elif ts - self.ts_prev > self.inactive_timeout or ts - self.ts_init > self.active_timeout:
            rtn_fv = self.get_fv()
            self.clear()
            self.ts_init = ts
        self.ps_buff.append(ps)
        self.dr_buff.append(dr)
        if self.ts_prev != 0:
            self.iat_buff.append(ts - self.ts_prev)
        if self.ts_prev_fwd != 0 and dr == 0:
            self.iat_buff_fwd.append(ts - self.ts_prev_fwd)
        if self.ts_prev_bwd != 0 and dr == 1:
            self.iat_buff_bwd.append(ts - self.ts_prev_bwd)
        self.ts_prev = ts
        if dr == 0:
            self.ts_prev_fwd = ts
        else:
            self.ts_prev_bwd = ts
        return rtn_fv
    

def callback(p):
    if IP not in p:
        return
    if TCP not in p and UDP not in p:
        return
    if DNS in p or DHCP in p or NTP in p:
        return
    tp = (p[IP].src, p[IP].dst, p.sport, p.dport, p.proto)
    tp_reverse = (p[IP].dst, p[IP].src, p.dport, p.sport, p.proto)
    if tp in buff_dict:
        buff = buff_dict[tp]
        dr = 0
    elif tp_reverse in buff_dict:
        tp = tp_reverse
        buff = buff_dict[tp]
        dr = 1
    else:
        buff = FlowBuff(tp)
        buff_dict[tp] = buff
        dr = 0
    ts = float(p.time)
    ps = p[IP].len
    fv = buff.update_and_yield(ts, ps, dr)
    if len(fv):
        print(f'new sample from {buff.tp}')
        X.append(list(tp) + fv)


def finalize():
    """
    在最后，处理那些未发生timeout、但确实提取了一些包特征的流，
    很可能是从未达到过active_timeout的短流
    """
    for tp in buff_dict:
        buff = buff_dict[tp]
        if buff.get_count():
            fv = buff.get_fv()
            print(f'new sample from {buff.tp}')
            X.append(list(tp) + fv)


def cicids_process(subset):
    df = pd.read_csv(CICIDS_TIME)
    """
    这里我使用了一个全局变量的dict: attack_info来存攻击事件的相关信息
    因为大部分攻击都来自于一小部分ip地址，所以选用了攻击者ip作为key
    对于TON-IoT数据集, 因为它给出了详细的攻击五元组以及攻击时间，
    可以考虑使用五元组作为key，(攻击时间，攻击类别)的list作为value，提高标注效率
    category，sub_cat是攻击大类和小类(CICIDS数据集中给定的)，
    对于TON-IoT数据集，只标注category即可(文件名)
    """
    for _, row in df[df['weekday'] == subset].iterrows():
        h_start, m_start = row['start'].split(':')
        ts_start = datetime(2017, 7, int(row['day']), int(h_start), int(m_start)) + timedelta(hours=11)
        ts_start = ts_start.timestamp()
        h_end, m_end = row['end'].split(':')
        ts_end = datetime(2017, 7, int(row['day']), int(h_end), int(m_end)) + timedelta(hours=11)
        ts_end = ts_end.timestamp()
        attacker_ip, cat, sub_cat = row['attacker_ip'], row['category'], row['sub_cat']
        if attacker_ip not in attack_info:
            attack_info[attacker_ip] = []
        attack_info[attacker_ip].append((ts_start, ts_end, cat, sub_cat))

    pcap_file = os.path.join(CICIDS_PCAP_DIR, CICIDS_DICT[subset] + '.pcap')
    sniff(offline=pcap_file, store=0, prn=callback)
    finalize()
    df = pd.DataFrame(X)
    df.columns = CUSTOM_TUPLE_COLS + CUSTOM_FEAT_COLS + CUSTOM_CAT_COLS + [CUSTOM_LABEL_COL]
    df.to_csv(f'../dataset/CICIDS-2017/{subset}.csv', index=False)


def toniot_process(subset):
    """
    TODO
    """
                             

if __name__ == '__main__':
    """
    dataset: 数据集名称，如cicids, toniot
    subset: pcap的名称，如Thursday, ddos
    """
    D = sys.argv[1]
    subset = sys.argv[2]
    if D == 'cicids':
        cicids_process(subset)
