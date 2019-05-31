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
    def __init__(self, source, dest, btime, reward, prereward,backQ, preforward_port,predest,back):
        self.dest = dest  # 终点节点
        self.source = source  # 源节点
        self.node = source  # 当前节点
        self.birth = btime  # 路由开始时间
        self.hops = 0  # 跳数
        self.qtime = timeit.default_timer()  # 进入节点排队的时间
        self.sizemax = 1542
        self.sizemin = 84
        self.size = random.randint(84, 1542)
        self.type = random.choice(['Tcp', 'Udp'])
        self.next = None  # 下一跳节点
        self.reward = reward
        self.prereward = prereward
        self.backQ = backQ
        self.forward_port = None  # 到下一跳节点的port
        self.preforward_port = preforward_port
        self.predest = predest
        self.back = back


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
        self.back = defaultdict(dict)  #储存学习返回内容

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
                self.back[i][self.links[i][j]] = Queue()
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
                packet.reward = current_time - packet.qtime
                packet.qtime = current_time
                self.packet_queue[next_hop].put(packet)
                self.packet_queue_size[next_hop].put(packet.size + next_hop_current_buffer)
                current_node_current_buffer = self.packet_queue_size[packet.node].get()
                current_node_changed_buffer = current_node_current_buffer - packet.size
                self.packet_queue_size[packet.node].put(current_node_changed_buffer)
                #print(next_hop,'forwarded')

    def _receiveinqueue(self, node):
        j = 1
        epsilon = 0.9
        while True:
            if self.packet_queue[node].empty():
                pass
            else:
                packet = self.packet_queue[node].get()
                time.sleep(self.process)
                current_time = timeit.default_timer()
                elapse_time = current_time - packet.birth
                if elapse_time > 70:
                    self.send_fail += 1
                    current_node_current_buffer = self.packet_queue_size[node].get()
                    current_node_changed_buffer = current_node_current_buffer - packet.size
                    self.packet_queue_size[node].put(current_node_changed_buffer)
                    #print(node,'time out')
                    continue
                dest = packet.dest
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
                    for i in range(5):
                        alt_sample_list = [0,1,2,3,4]
                        chosed_sample_index = np.random.choice(alt_sample_list,p=p.ravel())
                        self.sample[node].append(self.replay[node][chosed_sample_index])
                    self.agent[node].learn(self.sample[node])
                    self.replay[node] = []
                    j += 1
                    if j % 3 == 0 and j != 0:
                        #print(node,'update')
                        epsilon = self.agent[node].update_network(epsilon)
                if packet.next != dest and packet.next != None:  # 地址不相同且下一跳不是空，表示正常的包  #动作选择
                    prenode = packet.node
                    preforward_port = packet.forward_port
                    packet.node = packet.next
                    reward = packet.reward
                    prereward = packet.prereward
                    predest = packet.predest
                    receive_port = self.port[prenode][preforward_port]
                    MinQ_port_eval, forward_port, Q_estimate_list = self.agent[node].estimate(dest,receive_port,epsilon)
                    packet.forward_port = forward_port
                    next_hop = self.links[node][forward_port]
                    MinQ = self.agent[node].target(dest, MinQ_port_eval)           # 价值评估
                    self.back[node][prenode].put((reward,MinQ,preforward_port,dest))

                    if packet.back:
                        f_port = self.port[prenode][preforward_port]
                        backQ = packet.backQ
                        Q_actual = prereward + 0.9 * backQ
                        TD_target = Q_actual - min(Q_estimate_list)
                        weight = abs(TD_target)
                        sample = [predest, f_port, prereward, Q_estimate_list, backQ, weight]
                        self.replay[node].append(sample)

                    if not self.back[node][next_hop].empty():
                        back_infor = self.back[node][next_hop].get()
                        packet.prereward = back_infor[0]
                        packet.backQ = back_infor[1]
                        packet.preforward_port = back_infor[2]
                        packet.predest = back_infor[3]
                        packet.back = True
                    else:
                        packet.prereward = None
                        packet.backQ = None
                        packet.preforward_port = None
                        packet.predest = None
                        packet.back = False
                    packet.next = next_hop
                    self.packet_forward_queue[node][packet.forward_port].put(packet)
                    continue
                if packet.next == dest:  # 地址相同，但不是学习返回包，是routed包
                    prenode = packet.node
                    preforward_port = packet.forward_port
                    predest = packet.predest
                    packet.node = packet.next
                    forward_port = 0
                    packet.forward_port = forward_port
                    reward = packet.reward
                    prereward = packet.prereward
                    receive_port = self.port[prenode][preforward_port]
                    MinQ = 0  # 价值评估
                    self.back[node][prenode].put((reward,MinQ,preforward_port,dest))

                    if packet.back:
                        MinQ_port_eval, forward_port, Q_estimate_list = self.agent[node].estimate(dest,receive_port,epsilon)
                        f_port = self.port[prenode][preforward_port]
                        backQ = packet.backQ
                        Q_actual = prereward + 0.9 * backQ
                        TD_target = Q_actual - min(Q_estimate_list)
                        weight = abs(TD_target)
                        sample = [predest, f_port, prereward, Q_estimate_list, backQ, weight]
                        self.replay[node].append(sample)
                    self.packet_forward_queue[node][packet.forward_port].put(packet)
                    continue
                if packet.next == None:  # 新的发生包
                    receive_port = 0
                    MinQ_port_eval, forward_port, Q_estimate_list = self.agent[node].estimate(dest,receive_port,epsilon)
                    packet.forward_port = forward_port
                    next_hop = self.links[node][forward_port]
                    if not self.back[node][next_hop].empty():
                        back_infor = self.back[node][next_hop].get()
                        packet.prereward = back_infor[0]
                        packet.backQ = back_infor[1]
                        packet.preforward_port = back_infor[2]
                        packet.predest = back_infor[3]
                        packet.back = True
                    packet.next = next_hop
                    self.packet_forward_queue[node][packet.forward_port].put(packet)

    def _forwardqueue(self, node, port):
        while True:
            if self.packet_forward_queue[node][port].empty():
                pass
            else:
                packet = self.packet_forward_queue[node][port].get()
                forward_port = packet.forward_port
                transdelay = packet.size / self.linkc[node][forward_port]
                time.sleep(transdelay)
                self._step(packet)


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
                    plt.axis([0, 75, 0, 70])
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
                    plt.axis([0, 15, 0, 70])
                    plt.plot(x, delay_ave)
                    plt.title('node ' + str(i + 1) + ' to node ' + str(j + 1) + ' average delay')
                    plt.show()

    def _get_new_packet(self, node):
        callmean = self.arrivalmean[node]
        j = 1
        n_new_list = []
        p_list = []
        for i in range(1, self.n_nodes+1):
            if i != node:
                p_list.append(1/(self.n_nodes-1))
            else:
                p_list.append(0)
        p = np.array(p_list)
        dest_list = []
        dest_temp = []
        for i in range(1, self.n_nodes+1):
            dest_list.append(i)
        for i in range(1, self.n_nodes + 1):
            if i != node:
                n_new_list.append(int((self.iteration / (self.n_nodes - 1))))
                dest_temp.append(i)
            else:
                n_new_list.append(0)
        dest_temp_1 = dest_temp.copy()
        while True:
            time.sleep(callmean)
            source = node
            while True:
                dest = np.random.choice(dest_list, p=p.ravel())
                if node != dest and n_new_list[dest - 1] != 0 and dest in dest_temp_1:
                    n_new_list[dest - 1] -= 1
                    dest_temp_1.remove(dest)
                    if not dest_temp_1:
                        dest_temp_1 = dest_temp.copy()
                    break
                #print('node new ', node)
                if sum(n_new_list) == 0:
                    break
            current_time = timeit.default_timer()
            reward = 0
            backQ = None
            preforward_port = None
            predest = None
            back = False
            prereward = None
            packet = Packet(source, dest, current_time, reward, prereward,backQ, preforward_port,predest,back)
            node_buffer = self.buffer[node]
            #print('node get',node)
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
            print(node,j)
            if j > self.iteration:
                break


if __name__ == '__main__':
    N = Network()
    N.reset('network_sample.csv', 500)
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
    print(N.back)

    N.router()