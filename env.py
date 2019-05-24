import numpy as np
from collections import defaultdict
import random
from queue import Queue
import pandas
import Agent
import time
import timeit
from interval import Interval
import threading
from scipy import stats
import matplotlib.pyplot as plt
import inspect
import ctypes


def _async_raise(thread_id, c):
    thread_id = ctypes.c_long(thread_id)
    if not inspect.isclass(c):
        c = type(c)
    R = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(c))
    if R == 0:
        raise ValueError("No thread found")
    elif R != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
        raise SystemError("Asyncexc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


class Packet:
    def __init__(self, source, dest, btime, RL = False,back=False, forward_port=None, reward=0, backQ=None, predest=None):
        self.dest = dest  # 终点节点
        self.source = source  # 源节点
        self.node = source  # 当前节点
        self.birth = btime  # 路由开始时间
        self.hops = 0  # 跳数
        self.qtime = timeit.default_timer()  # 进入节点排队的时间
        self.sizemax = 1542
        self.sizemin = 84
        if not back:
            self.size = random.randint(84, 1542)
            self.RLback = False
            self.next = None  # 下一跳节点
            self.type = random.choice(['Tcp', 'Udp'])
            self.backQ = backQ
            self.reward = 0
            self.forward_port = None  # 到下一跳节点的port
            self.predest = None
            self.RL = RL
        else:
            self.size = 86
            self.RLback = True
            self.next = dest  # 下一跳节点
            self.type = 'Udp'
            self.backQ = backQ
            self.reward = reward
            self.forward_port = forward_port  # 到下一跳节点的port
            self.predest = predest
            self.RL = RL


class Network():
    def __init__(self):
        self.done = False  # 模拟完成
        self.packet_queue = defaultdict(Queue)  # 包转发事件的队列
        self.packet_queue_size = defaultdict(Queue)  # 每个节点queue队列的大小
        self.packet_forward_queue = defaultdict(dict)  # 每个node的转发queue（只是处理好的包，和排队queue中的包共同占用buffer）
        self.n_receive_queue = defaultdict(Queue)  # 一个node接受排队的包的个数 字典 key是每一个node
        self.replay = defaultdict(list)  # 一个node的经验回放池
        self.sample = defaultdict(list)  # 一个node的一次采样
        self.n_forward_queue = defaultdict(Queue)  # 一个node转发排队的包的个数 字典 key是每一个node
        self.port = defaultdict(dict)  # port对应关系
        self.n_nodes = 0  # 节点个数
        self.n_edges = 0  # 路径个数
        self.links = defaultdict(dict)  # 嵌套字典 第一个KEY是node 对应的值是 这个node的所有action和做了这个action之后的下一跳节点 (网络拓扑)
        self.linkc = defaultdict(dict)  # 嵌套字典 第一个KEY是node 对应的值是 这个node的所有port的带宽
        self.buffer = defaultdict(int)  # 路由器queuing的最大值
        self.t_routing_time = 0.0  # 总路由事件
        self.succeed_packets = 0  # 成功转发的数据包个数（对于整个网络）
        self.t_hops = 0  # 整个网络的转发数
        self.process = 1  # 一个list储存所有node处理queue中一个包的时间
        self.active_packets = 0  # 整个网络里正在转发的包个数
        self.send_fail = 0  # 转发失败的次数
        self.arrivalmean = defaultdict(int)  # 泊松过程中的λ
        self.agent = defaultdict()  # 路由器
        self.iteration = 0  # 总迭代次数
        self.latency = defaultdict(dict)  # 定义delay

    def reset(self, name, iteration):
        self.iteration = iteration
        net_topo = pandas.read_csv(name, header=None)
        self.n_nodes = net_topo.shape[0]  # 节点个数
        for i in range(1, net_topo.shape[0] + 1):
            self.packet_queue[i] = Queue()  # 给每个节点初始化queue队列 每个queue都是空队列
            self.packet_queue_size[i] = Queue(1)  # 给每个节点初始话queue队列大小 每个queue的大小都是0
            self.n_receive_queue[i] = Queue(1)  # 初始化接收队列排队个数
            self.n_forward_queue[i] = Queue(1)  # 初始化转发队列排队个数
            self.replay[i] = []  # 初始化经验回放池
            self.sample[i] = []  # 初始化采样
        for i in range(1, len(self.packet_queue_size) + 1):
            self.packet_queue_size[i].put(0)
        for i in range(1, net_topo.shape[0] + 1):
            self.links[i][0] = i
            for j in range(net_topo.loc[i - 1, :].shape[0]):
                if net_topo.loc[i - 1, j] == 'N':
                    p = j + 1
                    port = 1
                    while True:
                        self.links[i][port] = int(net_topo.loc[i - 1, p])
                        port += 1
                        p += 1
                        if net_topo.loc[i - 1, p] == 'C':
                            break
                if net_topo.loc[i - 1, j] == 'C':
                    q = j + 1
                    port = 0
                    while True:
                        self.linkc[i][port] = float(net_topo.loc[i - 1, q])
                        self.packet_forward_queue[i][port] = Queue()
                        port += 1
                        q += 1
                        if net_topo.loc[i - 1, q] == 'B':
                            break
                if net_topo.loc[i - 1, j] == 'B':
                    self.buffer[i] = float(net_topo.loc[i - 1, j + 1])
                if net_topo.loc[i - 1, j] == 'M':
                    self.arrivalmean[i] = float(net_topo.loc[i - 1, j + 1])
        for i in range(1, self.n_nodes + 1):
            for j in range(1, len(self.links[i])):
                next_node = self.links[i][j]
                key = filter(lambda x: i == x[1], self.links[next_node].items())
                key = dict(key)
                mapped_port = list(key.keys())[0]
                self.port[i][j] = mapped_port
        for i in range(1, net_topo.shape[0] + 1):
            print(i)
            self.agent[i] = Agent.ADQN(self.n_nodes, i, len(self.links[i]) - 1)  # 初始化路由器Q网络
        for i in range(1, self.n_nodes + 1):
            for j in range(1, self.n_nodes + 1):
                if j != i:
                    self.latency[i][j] = []

    def _step(self, packet):
        port = packet.forward_port
        if port == 0:  # 终点节点是自己 port0表示传输给自己的内网
            #print(packet.node,'routed')
            self.succeed_packets += 1
            self.active_packets -= 1  # 整个网络里正在转发的包个数
            current_node_current_buffer = self.packet_queue_size[packet.node].get()
            current_node_changed_buffer = current_node_current_buffer - packet.size
            self.packet_queue_size[packet.node].put(current_node_changed_buffer)
            if not packet.RL:
                time = timeit.default_timer()
                delay = time - packet.birth
                self.latency[packet.source][packet.dest].append(delay)
        else:
            next_hop = packet.next
            next_hop_current_buffer = self.packet_queue_size[next_hop].get()
            if next_hop_current_buffer + packet.size > self.buffer[next_hop]:
                self.send_fail += 1
                self.packet_queue_size[next_hop].put(next_hop_current_buffer)
                current_node_current_buffer = self.packet_queue_size[packet.node].get()
                current_node_changed_buffer = current_node_current_buffer - packet.size
                self.packet_queue_size[packet.node].put(current_node_changed_buffer)
                #print(next_hop,'full')
            else:                                                                            #下一跳节点的部分功能在这里实现
                packet.hops += 1
                current_time = timeit.default_timer()
                if not packet.RLback:
                    packet.reward = current_time - packet.qtime
                packet.qtime = current_time
                self.packet_queue[next_hop].put(packet)
                self.packet_queue_size[next_hop].put(packet.size + next_hop_current_buffer)
                current_node_current_buffer = self.packet_queue_size[packet.node].get()
                current_node_changed_buffer = current_node_current_buffer - packet.size
                self.packet_queue_size[packet.node].put(current_node_changed_buffer)
                #print(next_hop,'forwarded')

    def _receiveinqueue(self, node):
        i = 0
        safe = 0
        while True:
            if self.packet_queue[node].empty():
                pass
            else:
                packet = self.packet_queue[node].get()
                time.sleep(self.process)
                current_time = timeit.default_timer()
                elapse_time = current_time - packet.birth
                if elapse_time > 50:
                    self.send_fail += 1
                    current_node_current_buffer = self.packet_queue_size[node].get()
                    current_node_changed_buffer = current_node_current_buffer - packet.size
                    self.packet_queue_size[node].put(current_node_changed_buffer)
                    #print(node,'time out')
                    continue
                dest = packet.dest
                if packet.next != dest and packet.next != None:  # 地址不相同且下一跳不是空，表示正常的包  #动作选择
                    prenode = packet.node
                    preforward_port = packet.forward_port
                    packet.node = packet.next
                    predest = dest
                    receive_port = self.port[prenode][preforward_port]
                    if packet.RL and not packet.RLback:
                        # if safe == 0:
                        #     MinQ_port_eval, forward_port, MinQ_eval = self.agent[node].estimate(dest, receive_port,update_weight=True)
                        #     safe = 1
                        #     #print('first save normal')
                        if i % 14 == 0:
                            MinQ_port_eval, forward_port, MinQ_eval = self.agent[node].estimate(dest,receive_port ,update_weight=True)
                            # print(node,'update network')
                        else:
                            MinQ_port_eval, forward_port, MinQ_eval = self.agent[node].estimate(dest,receive_port)
                        packet.forward_port = forward_port
                        next_hop = self.links[node][forward_port]
                        reward = packet.reward
                        MinQ, Q_min_target_action = self.agent[node].target(dest, receive_port, MinQ_port_eval)           # 价值评估
                        self._backq(node, prenode, reward, MinQ, preforward_port, predest)
                        #print('normal backq',node,packet.node,forward_port,packet.next ,prenode, preforward_port, packet.source,predest
                        #print(node, 'forward')
                    else:
                        neighbour = False
                        for i in range(1, len(self.links[node])):
                            if packet.dest == self.links[node][i]:
                                packet.forward_port = i
                                next_hop = self.links[node][packet.forward_port]
                                neighbour = True
                                break
                        if not neighbour:
                            MinQ, Q_min_target_action = self.agent[node].target(dest, receive_port ,1)
                            packet.forward_port = Q_min_target_action
                            next_hop = self.links[node][packet.forward_port]
                    packet.next = next_hop
                    self.packet_forward_queue[node][packet.forward_port].put(packet)
                    continue
                if not packet.RLback and packet.next == dest:  # 地址相同，但不是学习返回包，是routed包
                    prenode = packet.node
                    preforward_port = packet.forward_port
                    packet.node = packet.next
                    forward_port = 0
                    packet.forward_port = forward_port
                    reward = packet.reward
                    MinQ = 0  # 价值评估
                    predest = dest
                    if packet.RL:
                        self._backq(node, prenode, reward, MinQ, preforward_port, predest)  #print('routed backq',node,packet.node,forward_port,packet.next, prenode, preforward_port,packet.source ,predest)
                    self.packet_forward_queue[node][packet.forward_port].put(packet)
                    continue
                if packet.next == None:  # 新的发生包
                    # if safe == 0:
                    #     receive_port = 0
                    #     MinQ_port_eval, forward_port, MinQ_eval = self.agent[node].estimate(dest,receive_port,update_weight=True)
                    #     safe = 1
                        #print('first save new')
                    if not packet.RL:
                        neighbour = False
                        for i in range(1, len(self.links[node])):
                            if packet.dest == self.links[node][i]:
                                packet.forward_port = i
                                next_hop = self.links[node][packet.forward_port]
                                neighbour = True
                                break
                        if not neighbour:
                            receive_port = 0
                            MinQ_acutal, forward_port = self.agent[node].target(dest,receive_port,1)
                            #print(node,'update network')
                            packet.forward_port = forward_port
                            next_hop = self.links[node][forward_port]
                    else:
                        receive_port = 0
                        if i % 14 == 0:
                            MinQ_port_eval, forward_port, MinQ_eval = self.agent[node].estimate(dest, receive_port,update_weight=True)
                            # print(node,'update network')
                        else:
                            MinQ_port_eval, forward_port, MinQ_eval = self.agent[node].estimate(dest,receive_port)
                        packet.forward_port = forward_port
                        next_hop = self.links[node][forward_port]
                    packet.next = next_hop
                    #print('normal backq',node,packet.node,forward_port,packet.next ,prenode, preforward_port, packet.source,predest
                    self.packet_forward_queue[node][packet.forward_port].put(packet)
                    #print(node, 'forward')
                    continue
                    #print('new packet receive', node, packet.node, packet.next, forward_port, packet.source, packet.dest)
                if packet.next == dest and packet.RLback:  # 是返回的RL包，经验回放
                    #print(node,'RL bakc receive')
                    prenode = packet.node
                    preforward_port = packet.forward_port
                    predest = packet.predest
                    reward = packet.reward
                    f_port = self.port[prenode][preforward_port]
                    MinQ = packet.backQ
                    Q_actual = reward + 0.9 * MinQ
                    MinQ_port_eval, forward_port, Q_estimate_list = self.agent[node].estimate(predest,f_port)
                    TD_target = Q_actual - min(Q_estimate_list)
                    weight = abs(TD_target)
                    sample = [predest, f_port, reward, Q_estimate_list, MinQ, weight]
                    self.replay[node].append(sample)
                    current_node_current_buffer = self.packet_queue_size[node].get()
                    current_node_changed_buffer = current_node_current_buffer - packet.size
                    self.packet_queue_size[node].put(current_node_changed_buffer)
                    if len(self.replay[node]) == 5:
                        #print(node,'learn')
                        self.sample[node] = []
                        importance_sampling_factor = []
                        pr_list = []
                        TD_list = []
                        for i in range(5):
                            TD_list.append(self.replay[node][i][5])
                        TD_sum = sum(TD_list)
                        for i in range(5):
                            pr_list.append(self.replay[node][i][5]/TD_sum)
                        p = np.array(pr_list)
                        for i in range(5):
                            importance_sampling_factor.append((0.2/pr_list[i])**0.1)
                            self.replay[node][i].append(importance_sampling_factor[i])
                        for i in range(3):
                            alt_sample_list = [0,1,2,3,4]
                            chosed_sample_index = np.random.choice(alt_sample_list,p=p.ravel())
                            self.sample[node].append(self.replay[node][chosed_sample_index])
                        # interval_list = []
                        # for i in range(5):
                        #     if i == 0:
                        #         interval_list.append(Interval(0, self.replay[node][0][5], upper_closed=False))
                        #         temp = self.replay[node][0][5]
                        #     else:
                        #         interval_list.append(Interval(temp, temp + self.replay[node][i][5], upper_closed=False))
                        #         temp += self.replay[node][i][5]
                        # for i in range(3):
                        #     seed = random.uniform(0, interval_list[4].upper_bound)
                        #     for j in range(5):
                        #         if seed in interval_list[j]:
                        #             self.sample[node].append(self.replay[node][j])
                        self.agent[node].learn(self.sample[node])
                        self.replay[node] = []
            i += 1
            # if i % 100000 == 0:
            #     print('receive',node,i)
            # if i > self.iteration*2:
            #     break

    def _forwardqueue(self, node, port):
        i = 0
        while True:
            if self.packet_forward_queue[node][port].empty():
                pass
            else:
                packet = self.packet_forward_queue[node][port].get()
                forward_port = packet.forward_port
                transdelay = packet.size / self.linkc[node][forward_port]
                time.sleep(transdelay)
                self._step(packet)
            i += 1
            # if i % 100000 == 0:
            #     print('forward',node,port,i)
            # if i > self.iteration*2:
            #     break

    def _backq(self, node, prenode, reward, MinQ, preforward_port, predest):
        time = timeit.default_timer()
        f_port = self.port[prenode][preforward_port]
        RLback_packet = Packet(node, prenode, time,RL = True, back=True, forward_port=f_port, reward=reward, backQ=MinQ,predest=predest)
        current_node_current_buffer = self.packet_queue_size[node].get()
        if current_node_current_buffer + RLback_packet.size > self.buffer[node]:
            self.send_fail += 1
            self.packet_queue_size[node].put(current_node_current_buffer)
            #print(node, 'RL back generate fail')
        else:
            self.packet_queue_size[node].put(current_node_current_buffer + RLback_packet.size)
            #print(node,prenode,f_port,preforward_port)
            self.packet_forward_queue[node][f_port].put(RLback_packet)
            #print(node, 'RL back generate success')

    def router(self):
        rece_list = []
        forw_list = []
        get_new_packet_list = []
        for i in range(self.n_nodes):
            rece_list.append(threading.Thread(target=Network._receiveinqueue, args=(self, i + 1)))
        print(len(rece_list))
        for i in range(self.n_nodes):
            for j in range(len(self.links[i + 1])):
                forw_list.append(threading.Thread(target=Network._forwardqueue, args=(self, i + 1, j)))
        for i in range(self.n_nodes):
            get_new_packet_list.append(threading.Thread(target=Network._get_new_packet, args=(self, i + 1)))
        for i in range(self.n_nodes):
            rece_list[i].start()
        for i in range(len(forw_list)):
            forw_list[i].start()
        for i in range(self.n_nodes):
            get_new_packet_list[i].start()
        # for i in range(self.n_nodes):
        #     rece_list[i].join()
        # for i in range(self.n_nodes):
        #     forw_list[i].join()
        for i in range(self.n_nodes):
            get_new_packet_list[i].join()
        for i in range(self.n_nodes):
            stop_thread(rece_list[i])
        for i in range(len(forw_list)):
            stop_thread(forw_list[i])
        print(self.latency)
        for i in range(len(self.latency)):
            for j in range(len(self.latency[i + 1])+1):
                if j + 1 != i + 1:
                    x_list_len = len(self.latency[i + 1][j + 1])
                    x = []
                    for q in range(x_list_len):
                        x.append(q)
                    plt.plot(x, self.latency[i + 1][j + 1])
                    plt.title('node ' + str(i + 1) + ' to node ' + str(j + 1) + ' delay')
                    plt.show()
        for i in range(len(self.latency)):
            for j in range(len(self.latency[i + 1]) + 1):
                if j + 1 != i + 1:
                    x_list_len = int(len(self.latency[i + 1][j + 1]) / 5)
                    delay_ave = []
                    x = []
                    iter = 0
                    for q in range(x_list_len):
                        x.append(q)
                        lower_bound = q + iter * 4
                        upper_bound = q + (iter + 1) * 4
                        delay_ave.append(sum(self.latency[i + 1][j + 1][lower_bound:upper_bound]) / 5)
                        iter += 1
                    plt.axis([0, 10, 0, 30])
                    plt.plot(x, delay_ave)
                    plt.title('node ' + str(i + 1) + ' to node ' + str(j + 1) + ' average delay')
                    plt.show()

    def _get_new_packet(self, node):
        callmean = self.arrivalmean[node]
        j = 1
        n_new_list = []
        n_train_list = []
        p_list = []
        for i in range(1, self.n_nodes+1):
            if i != node:
                p_list.append(1/(self.n_nodes-1))
            else:
                p_list.append(0)
        p = np.array(p_list)
        dest_list = []
        for i in range(1, self.n_nodes+1):
            dest_list.append(i)
        for i in range(1, self.n_nodes+1):
            if i != node:
                n_new_list.append(int((self.iteration / (self.n_nodes-1)) * 0.25))
            else:
                n_new_list.append(0)
        for i in range(1, self.n_nodes+ 1):
            if i != node:
                n_train_list.append(int((self.iteration / (self.n_nodes-1)) * 0.75))
            else:
                n_train_list.append(0)
        while True:
            arrival = stats.expon.rvs(scale=1 / callmean, size=1)
            if arrival < 20 and arrival > 5:
                break
        while True:
            time.sleep(arrival)
            source = node
            if j % 4 != 0:
                while True:
                    dest = np.random.choice(dest_list, p=p.ravel())
                    if node != dest and n_train_list[dest - 1] != 0:
                        RL = False
                        n_train_list[dest - 1] -= 1
                        break
                    if sum(n_train_list) == 0:
                        break
            else:
                while True:
                    dest = np.random.choice(dest_list, p=p.ravel())
                    if node != dest and n_new_list[dest - 1] != 0:
                        RL = True
                        n_new_list[dest - 1] -= 1
                        break
                    if sum(n_new_list) == 0:
                        break
            current_time = timeit.default_timer()
            packet = Packet(source, dest, current_time,RL)
            node_buffer = self.buffer[node]
            current_buffer = self.packet_queue_size[node].get()
            if current_buffer + packet.size > node_buffer:
                self.send_fail += 1
                self.packet_queue_size[node].put(current_buffer)
                #print('new', node, i, arrival * 10, 'new packet fail')
            else:
                self.active_packets += 1
                self.packet_queue[node].put(packet)
                self.packet_queue_size[node].put(packet.size + current_buffer)
                #print('new', node, i, arrival * 10)
            if j % 50 == 0 or j == 1:
                print(node, j, 'new done')
            j += 1
            if j > self.iteration:
                break


if __name__ == '__main__':
    N = Network()
    N.reset('network_sample.csv', 200)
    print(N.n_nodes)
    print(N.packet_queue)  # 给每个节点初始化queue队列 每个queue都是空队列
    print(N.packet_forward_queue)
    print(N.packet_queue_size)  # 给每个节点初始话queue队列大小 每个queue的大小都是0
    print(N.links)  # 嵌套字典 第一个KEY是node 对应的值是 这个node的所有action和做了这个action之后的下一跳节点 (网络拓扑)
    print(N.linkc)  # 嵌套字典 第一个KEY是node 对应的值是 这个node的所有port的带宽
    print(N.buffer)  # 路由器queuing的最大值
    print(N.arrivalmean)  # 泊松过程中的λ
    print(N.agent)
    print(N.port)
    print(N.latency)

    N.router()