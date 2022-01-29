import new_partitioing_graph
import partition_graph
import random
import collections

# G = partition_graph.build_karate_club_graph()
xadj = []
partition_sets = partition_graph.new_partitioing_graph
output = dict()
proxy_node_set = set()

# k = number of neighbors to sample
# G = FullGraph
# partition_sets = Store partition data
# sample_count = for every target node it will sample 15 sets of each containing 2 neighbors of target node
inter_avg = []
diff_avg = []
p = []
q = []
pp = []
flat_list = []

def sampling_function(G, alpha, k, partition_sets, xadj, adjncy, sample_count=15):
    print("Nodes of Graph", G.number_of_nodes())
    ##for node in range(len(G.nodes)):
    for node in range(G.number_of_nodes()):
        neighbors_set = adjncy[xadj[node]:(xadj[node + 1])]
        # print()
        # print("Neighbors of node {} are".format(node), neighbors_set)

        node_in_partition_set = None

        for partition in partition_sets:
            if node in partition:
                node_in_partition_set = partition
                break

        inter = node_in_partition_set.intersection(neighbors_set)
        x = len(inter)
        diff = set(neighbors_set) - inter
        y = len(diff)

        # print("Own partition neighbors of node {} is:-".format(node), inter)
        inter_avg.append(len(inter))
        # print("Different partition neighbors of node {} is:-".format(node), diff)
        diff_avg.append(len(diff))

        w1 = []
        w2 = []
        for i in range(x):
            w1.append(alpha)
            # print("w1", alpha)

        for i in range(y):
            w2.append(1)
            # print("w2", 1)

        sample_picks = []
        for i in range(sample_count):
            # print("population", list(inter) + list(diff))
            # print("weights", list(w1) + list(w2))
            sample_pick = random.choices(population=list(inter) + list(diff), weights=list(w1) + list(w2), k=k)
            sample_picks.append(sample_pick)
            # sample_pick = random.choices(population=list(inter) + list(diff), k=k)
            # number of node which are in same partition
            own_partition_count = 0
            for i in sample_pick:
                if i in node_in_partition_set:
                    own_partition_count += 1

            # Proxy nodes are those node which are from different partition set
            # and less than own partition set
            if own_partition_count > 0 and own_partition_count < k:
                # print("Edge cut",sample_pick)
                proxy_node_set.add(frozenset(sample_pick))

            if node in output:
                output[node].append(sample_pick)
            else:
                output[node] = [sample_pick]

        flat_list = [item for sublist in sample_picks for item in sublist]

        # print("neighbors node after sampling", flat_list)

        def intersection(A, B):
            c_a = collections.Counter(A)
            c_b = collections.Counter(B)
            # print("Key value of node and number of times occur after sampling", c_a)
            # print("neighbors Nodes in same partition", c_b)
            # print("keys of total sampled list", c_a.keys())
            # print("keys of node in same partition", c_b.keys())

            a = c_a.keys()
            b = c_b.keys()
            c = a - b
            pp.append(c)

            # print("Proxy Node is", c)
            duplicates = []
            for c in c_a:
                duplicates += [c] * min(c_a[c], c_b[c])
                different = c_a.keys() - duplicates

        #     if len(different) == 0:
        #         print("q is empty", 0)
        #         print("p is", len(flat_list))
        #     else:
        #         q_count = []
        #         for i in set(diff):
        #             q_count.append(c_a.get(i))
        #         for g in flat_list:
        #             if g in c_a.keys() & c_b.keys():
        #                 (c_a.keys() & c_b.keys()).remove(g)
        #
        #         print("q is", sum(filter(None, list(q_count))))
        #         print("p is", len(flat_list) - sum(filter(None, list(q_count))))
        #     return duplicates
        #     print()
        #
        # intersection(flat_list, inter)

        # own_partition = set(flat_list).intersection(inter)
        # own_partition = remove_duplicate(flat_list,inter)    #testing
        # print("own partition", own_partition)
        # p.append(len(own_partition))
        # print("p is :",p)
        # diff_partition = set(flat_list).intersection(diff)
        # #diff_partition = remove_duplicate(flat_list, diff)
        #
        # print("other partition", diff_partition)
        # q.append(len(diff_partition))
        # print("q is :", q)

    # print("list of inter nodes", list(inter))
    # print("list of other nodes", list(diff))

    # for node in output:
    #     print("Sample set for target node {} is {}".format(node, output[node]))
    # print()

    # for i in proxy_node_set:
    #     print(list(i))

    # print("total number of edge cut", len(proxy_node_set))
    flatten1 = [element for items in proxy_node_set for element in items]

    # print("Proxy node", set(flatten1))
    # print("number of proxy node", len(set(flatten1)))
    Proxy_node_flatten = [element for items in pp for element in items]
    # print("New Proxy node", set(Proxy_node_flatten))
    print("Length of new proxy nodes", len(set(Proxy_node_flatten)))


def optimized_sampling_function(G, alpha, k, partition_sets, xadj, adjncy, sample_count=15):
    D_list = []

    for p in range(len(partition_sets)):
        D_list.append(set())

    # print("Dlist is",D_list)
    for node in range(G.number_of_nodes()):
        neighbors_set = adjncy[xadj[node]:(xadj[node + 1])]
        # print()
        # print("Neighbors of node {} are".format(node), neighbors_set)

        node_in_partition_set = None
        same_partition_set_index = None

        for partition_index in range(len(partition_sets)):
            if node in partition_sets[partition_index]:
                node_in_partition_set = partition_sets[partition_index]
                same_partition_set_index = partition_index
                break

        same_partition_dynamic_set = D_list[same_partition_set_index]
        same_partition_dynamic_size = len(same_partition_dynamic_set)

        inter = node_in_partition_set.intersection(neighbors_set)
        x = len(inter)

        diff = set(neighbors_set) - inter
        y = len(diff)
        # print("-------------------------------")
        # print("Own partition neighbors of node {} is:-".format(node), inter)
        # inter_avg.append(len(inter))
        # print("Different partition neighbors of node {} is:-".format(node), diff)
        # diff_avg.append(len(diff))
        # print("X is ", x)
        # print("Y is", y)
        # print("Size of Dynamic list", same_partition_dynamic_size)
        # print("D list", D_list)
        # print("proxy length of optimized", len(D_list[0]) + len(D_list[1]))
        w1 = []
        w2 = []

        for i in range(x):
            w1.append(alpha / 10 / (x + same_partition_dynamic_size))

        for i in range(y):
            w2.append((alpha / 10 / (x + same_partition_dynamic_size)) / (len(neighbors_set)))
            # w2.append((1 - alpha/10)/(len(neighbors_set) + y))
            # w2.append((1 - alpha/10)/y)

        # print("weights for node {} is {}".format(node, list(w1) + list(w2)))
        sample_picks = []
        for i in range(sample_count):
            # print("population", list(inter) + list(diff))
            # print("weights", list(w1) + list(w2))
            sample_pick = random.choices(population=list(inter) + list(diff), weights=list(w1) + list(w2), k=k)
            sample_picks.append(sample_pick)

            for sampled_node in sample_pick:
                if sampled_node not in node_in_partition_set:
                    D_list[same_partition_set_index].add(sampled_node)
                # else:
                #     for partition_index in range(len(partition_sets)):
                #         if sampled_node in partition_sets[partition_index]:
                #             diff_partition_index = partition_index
                #             break
                #     # find partition set in which sample node belongs to
                #     # find partition index of this partition = diff_partition_index
                #     D_list[diff_partition_index].add(sampled_node)
            # sample_pick = random.choices(population=list(inter) + list(diff), k=k)
            # number of node which are in same partition
            own_partition_count = 0
            for i in sample_pick:
                if i in node_in_partition_set:
                    own_partition_count += 1

            if node in output:
                output[node].append(sample_pick)
            else:
                output[node] = [sample_pick]

        flat_list = [item for sublist in sample_picks for item in sublist]

        # print("neighbors", neighbors_set)
        # print("sample picked", sample_picks)
        # print("neighbors node after sampling", flat_list)

        def intersection(A, B):
            c_a = collections.Counter(A)
            c_b = collections.Counter(B)
            # print("Key value of node and number of times occur after sampling", c_a)
            # print("neighbors Nodes in same partition", c_b)
            # print("keys of total sampled list", c_a.keys())
            # print("keys of node in same partition", c_b.keys())

            a = c_a.keys()
            b = c_b.keys()
            c = a - b
            pp.append(c)

            # print("Proxy Node is", c)
            duplicates = []
            for c in c_a:
                duplicates += [c] * min(c_a[c], c_b[c])
                different = c_a.keys() - duplicates

            # if len(different) == 0:
            #     print("q is empty", 0)
            #     print("p is", len(flat_list))
            # else:
            #     q_count = []
            #     for i in set(diff):
            #         q_count.append(c_a.get(i))
            #     for g in flat_list:
            #         if g in c_a.keys() & c_b.keys():
            #             (c_a.keys() & c_b.keys()).remove(g)
            #
            #     print("q is", sum(filter(None, list(q_count))))
            #     print("p is", len(flat_list) - sum(filter(None, list(q_count))))
            # return duplicates
            # print()

        intersection(flat_list, inter)

        # own_partition = set(flat_list).intersection(inter)
        # own_partition = remove_duplicate(flat_list,inter)    #testing
        # print("own partition", own_partition)
        # p.append(len(own_partition))
        # print("p is :",p)
        # diff_partition = set(flat_list).intersection(diff)
        # #diff_partition = remove_duplicate(flat_list, diff)
        #
        # print("other partition", diff_partition)
        # q.append(len(diff_partition))
        # print("q is :", q)

    print()
    # print("list of inter nodes", list(inter))
    # print("list of other nodes", list(diff))
    # print()

    # print()
    # for node in output:
    #     print("Sample set for target node {} is {}".format(node, output[node]))
    # print()

    # for i in proxy_node_set:
    #     print(list(i))

    # print()
    # print("total number of edge cut", len(proxy_node_set))
    flatten1 = [element for items in proxy_node_set for element in items]

    # print("Proxy node", set(flatten1))
    # print("number of proxy node", len(set(flatten1)))
    Proxy_node_flatten = [element for items in pp for element in items]
    # print("New Proxy node", set(Proxy_node_flatten))
    # print("Length of optimized proxy nodes", len(D_list))
    # print("Length of optimized proxy nodes", len(set(Proxy_node_flatten)))
    print("proxy length of optimized", len(D_list[0]) + len(D_list[1]))





# sampling_function(G, 10, 15, partition_sets, xadj, adjncy, sample_count=15)
# optimized_sampling_function(G, 10, 15, partition_sets, xadj, adjncy, sample_count=15)