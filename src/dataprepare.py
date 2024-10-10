# -*- encoding: utf-8 -*-
'''
Created on 2017年5月5日

@author: Baijie
'''
try:
    import xml.etree.cElementTree as ET
except ImportError:
    print('cElementTree unavailable!')
    import xml.etree.ElementTree as ET
import sys, re, os
import pandas as pd


def edges_for_node2vec(data, output_path):
    node_count = 0
    node_dict = {}
    edges = set()
    t = 0
    for pub, title, authors in data:
        for author in authors:
            if author not in node_dict:
                node_dict[author] = node_count
                node_count += 1
        edges.update([(node_dict[x], node_dict[y]) for x in authors for y in authors if x != y])
        #         edges.update([(node_dict[authors[x]],node_dict[authors[y]]) for x in range(len(authors)-1) for y in range(x+1,len(authors))])
        t += 1
        if t % 10000 == 0:
            print(t)
    write_dicts(node_dict, '../model/' + data.dataname + '_node_dict.txt')
    write_edges(edges, output_path)
    print('Finish.')


def read_node_labels(data, label_size):
    group_nodes = {}
    for group, item, nodes in data:
        if group in group_nodes:
            group_nodes[group] += len(nodes)
        else:
            group_nodes[group] = len(nodes)
    if label_size > len(group_nodes):
        print('This dataset do not have so much groups.')
        return None
    else:
        group_nodes = sorted(group_nodes.items(), key=lambda item: item[1], reverse=True)
        popular_groups = [g[0] for g in group_nodes[:label_size]]
        node_labels = {}
        for group, item, nodes in data:
            if group in popular_groups:
                for node in nodes:
                    if node in node_labels:
                        t_labels = node_labels[node]
                    else:
                        t_labels = [0] * label_size
                        node_labels[node] = t_labels
                    t_labels[popular_groups.index(group)] = 1

        return node_labels


def read_item_labels(data, label_size):
    popular_pubs = []
    for i in list(data.collect_labels())[:label_size]:
        popular_pubs.append(i[0])
    #     print popular_pubs

    #     node_dict = read_dicts(data.nodedictpath)

    label_nodes = []
    for pub, title, authors in data:
        if pub in popular_pubs:
            label = popular_pubs.index(pub)
            #             t_nodes = []
            #             for author in authors:
            #                 t_nodes.append(author)
            label_nodes.append((label, authors))

    return label_nodes


def read_item_labels_h(data, label_size):
    popular_pubs = []
    for i in list(data.collect_labels())[:label_size]:
        popular_pubs.append(i[0])
    #     print popular_pubs

    #     node_dict = read_dicts(data.nodedictpath)

    label_nodes = []
    for pub, title, authors in data:
        if pub in popular_pubs:
            label = popular_pubs.index(pub)
            label_nodes.append((label, title))
    print('items: ' + str(len(label_nodes)))
    return label_nodes


def read_item_nodes(data):
    node_dict = read_dicts(data.nodedictpath)

    item_nodes = []
    for pub, title, authors in data:
        t_nodes = []
        for author in authors:
            author_id = node_dict[author]
            t_nodes.append(author_id)
        item_nodes.append(t_nodes)
    node_list = [v for (k, v) in node_dict.items()]
    return item_nodes, node_list


def read_item_nodes_h(data):
    item_nodes = {}
    for pub, title, authors in data:
        item_nodes[title] = authors
    return item_nodes


class Wiki(object):

    def __init__(self, path='../data/wikipedia.tsv'):
        self.dataname = 'wiki'
        self.path = path
        self.isdirected = False
        self.isheterogeneous = False
        self.group_nums = 15


    def __iter__(self):
        try:
            fread = open(self.path)
        except:
            print('Error:cannot parse file:' + self.path)
            sys.exit(1)

        for line in fread:
            if line[0] != '#':
                line = line.strip().split('\t')
                if len(line) == 2:
                    yield line[0].strip(), line[1].strip()

    def collect_labels(self):
        fread = open('../data/wikicat.tsv', 'r')
        group_dict = {}
        for line in fread:
            if len(line) >= 5 and line[0] != '#':
                line = line.strip().split('\t')
                group = line[1].split('.')[1]
                group_dict[group] = group_dict.setdefault(group, []) + [line[0]]
        return sorted(group_dict.items(), key=lambda item: len(item[1]), reverse=True)


class Wiki_hirarchy(object):
    def __init__(self, path='../data/wikicat.tsv'):
        self.dataname = 'wiki_hi'
        self.path = path
        self.isdirected = False
        self.isheterogeneous = False

    def __iter__(self):
        try:
            fread = open(self.path)
        except:
            print('Error:cannot parse file:' + self.path)
            sys.exit(1)

        for line in fread:
            if len(line) >= 5 and line[0] != '#':
                line = line.strip().split('\t')
                groups = line[1].split('.')
                for (i, g) in enumerate(groups):
                    if i < len(groups) - 1:
                        yield g, groups[i + 1]

    def collect_labels(self):
        fread = open(self.path, 'r')
        group_dict = {0: ['subject']}
        for line in fread:
            if len(line) >= 5 and line[0] != '#':
                line = line.strip().split('\t')
                group = line[1].split('.')
                for i in range(1, len(group)):
                    label = group[1]
                    node_set = group_dict.setdefault(label, set())
                    node_set.add(group[i])
                    group_dict[label] = node_set
        #                     group_dict[label] = group_dict.setdefault(label,[]) + [group[i]]
        return dict([(k, list(v)) for (k, v) in group_dict.items()])

    def get_hierarchy(self):
        hierarchy = dict()  # hierarchy, for colors
        # construct hierarchies
        fread = open(self.path, 'r')
        for line in fread:
            if len(line) >= 5 and line[0] != '#':
                line = line.strip().split('\t')
                groups = line[1].split('.')
                for (i, g) in enumerate(groups):
                    hierarchy[g] = i
        fread.close()
        return hierarchy

class DIP(object):
    def __init__(self, path='../data/dip20170205.txt', nodedictpath=None):
        self.dataname = 'dip'
        self.path = path
        self.isdirected = False
        self.isheterogeneous = False
        self.group_nums = 15

    def extract_reg(self, reg, string):
        match = re.search(reg, string)
        if match:
            return match.group(1)
        else:
            return None

    def __iter__(self):
        dipreg = re.compile(r'DIP-(\d+)N')
        fread = open(self.path, 'r')
        for line in fread:
            line = line.split('\t')
            node1 = self.extract_reg(dipreg, line[0])
            node2 = self.extract_reg(dipreg, line[1])
            if node1 and node2:
                yield node1, node2

    def collect_labels(self):
        def add_dict(node, type):
            node_set = group_dict.setdefault(type, set())
            node_set.add(node)

        group_dict = {}
        typereg = re.compile(r'taxid:(\d+)\(.*\)')
        dipreg = re.compile(r'DIP-(\d+)N')
        fread = open(self.path, 'r')
        for line in fread:
            line = line.split('\t')
            node1 = self.extract_reg(dipreg, line[0])
            node2 = self.extract_reg(dipreg, line[1])
            type1 = self.extract_reg(typereg, line[9])
            type2 = self.extract_reg(typereg, line[10])
            if node1 and type1:
                add_dict(node1, type1)
            if node2 and type2:
                add_dict(node2, type2)
        return sorted(group_dict.items(), key=lambda item: len(item[1]), reverse=True)


class WITS(object):
    def __init__(self, path='../data/wits_en_trade'):
        self.dataname = 'wits'
        self.path = path
        self.isdirected = True
        self.isheterogeneous = False
        self.group_nums = 7

    def __iter__(self):
        list_file = os.listdir(self.path)
        for i in range(0, len(list_file)):
            path = os.path.join(self.path, list_file[i])
            if os.path.isfile(path):
                df = pd.read_csv(path, encoding='gbk')
                reporter = df['Reporter'].iloc[0]
                if isinstance(reporter, str):
                    new_df_export = df[df['Indicator'].isin(['Trade (US$ Mil)-Top 5 Export Partner'])]
                    new_df_inport = df[df['Indicator'].isin(['Trade (US$ Mil)-Top 5 Import Partner'])]
                    partners_out = set(new_df_export['Partner'])
                    for partner in partners_out:
                        if partner and partner != 'Unspecified':
                            yield reporter, partner
                    partners_in = set(new_df_inport['Partner'])
                    for partner in partners_in:
                        if partner and partner != 'Unspecified':
                            yield partner.strip(), reporter.strip()

    def collect_labels(self):
        def add_dict(node, type):
            node_set = group_dict.setdefault(type, set())
            node_set.add(node)

        group_dict = {}
        df = pd.read_excel('../data/WITSCountryProfile.xls')
        country_cat = list(df[['Country Name', 'Region']].itertuples(index=False))
        for country, cat in country_cat:
            if country and isinstance(cat, str):
                add_dict(country, cat)
        return sorted(group_dict.items(), key=lambda item: len(item[1]), reverse=True)

'''统一规定输出： 类别， item，node列表'''


class DBLP(object):
    def __init__(self, path='../data/dblp-2017-04-04.xml'):
        self.dataname = 'dblp'
        self.path = path
        self.isdirected = False
        self.isheterogeneous = True
        self.group_nums = 15

    def __iter__(self):
        try:
            tree = ET.parse(self.path)  # 打开xml文档
            root = tree.getroot()  # 获得root节点
        except:
            print('Error:cannot parse file:' + self.path)
            sys.exit(1)

        for item in root:
            try:
                title = item.find('title').text
                authors = []
                for author in item.findall('author'):
                    authors.append(author.text)
                if len(authors) <= 5 or len(authors) >= 50:
                    continue
                pub = item.find('journal')
                if pub == None:
                    pub = item.find('booktitle')
                if pub == None:
                    continue
                pub = pub.text
                yield pub, title, authors

            except:
                #                 print e
                continue

    def collect_labels(self):
        try:
            tree = ET.parse(self.path)  # 打开xml文档
            root = tree.getroot()  # 获得root节点
        except:
            print('Error:cannot parse file:' + self.path)
            sys.exit(1)
        pub_dict = {}
        for item in root:
            try:
                pub = item.find('journal')
                if pub == None:
                    pub = item.find('booktitle')
                if pub != None:
                    pub = pub.text
                    pub_dict[pub] = pub_dict.setdefault(pub, 0) + 1
            except:
                #                 print e
                continue

        return sorted(pub_dict.items(), key=lambda item: item[1], reverse=True)


class Amazon(object):
    #     670647 nodes and 7421621 edges when max num is 30
    def __init__(self, path='../data/amazon-meta.txt', n_min=5, n_max=50):
        self.dataname = 'amazon'
        self.path = path
        self.n_min = n_min
        self.n_max = n_max
        if n_max == None:
            self.n_max = float('inf')
        self.isdirected = False
        self.isheterogeneous = True
        self.group_nums = 7

    def __iter__(self):
        fread = open(self.path, 'r')
        flag_yield = False
        while True:
            try:
                line = fread.readline()
                if line == '':
                    fread.close()
                    break
                line = line.strip()
                if line == '' or line[0] == '#':
                    continue
                if line[:3] == 'Id:':
                    flag_yield = False
                    pid = line[3:].strip()
                    continue
                if line[:6] == 'group:':
                    group = line[6:].strip()
                    continue
                if line[:8] == 'reviews:':
                    line = line[8:].strip().split()
                    r_num = int(line[3].strip())
                    if r_num < self.n_min:
                        continue
                    elif r_num > self.n_max:
                        continue
                    else:
                        users = []
                        for i in range(r_num):
                            line = fread.readline().strip().split()
                            uid = line[2].strip()
                            users.append(uid)
                        flag_yield = True

                    if flag_yield == True:
                        yield group, pid, users
            except:
                break

    def collect_labels(self):
        fread = open(self.path, 'r')
        group_dict = {}
        while True:
            try:
                line = fread.readline()
                if line == '':
                    fread.close()
                    break
                line = line.strip()
                if line[:6] == 'group:':
                    group = line[6:].strip()
                if line[:8] == 'reviews:':
                    line = line[8:].strip().split()
                    r_num = int(line[3].strip())
                    if r_num < self.n_min:
                        continue
                    elif r_num > self.n_max:
                        continue
                    else:
                        group_dict[group] = group_dict.setdefault(group, 0) + 1
            except:
                break
        return sorted(group_dict.items(), key=lambda item: item[1], reverse=True)

    def get_node_names(self):  # node_names: key: id, value: title(string)
        fread = open(self.path, 'r')
        node_names = {}
        while True:
            try:
                line = fread.readline()
                if line == '':
                    fread.close()
                    return node_names
                line = line.strip()
                if line == '' or line[0] == '#':
                    continue
                if line[:3] == 'Id:':
                    pid = line[3:].strip()
                    continue
                if line[:6] == 'title:':
                    name = line[6:].strip()
                    node_names[pid] = name
                    continue
            except:
                return node_names


def write_dicts(dic, path):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fwrite = open(path, 'w')
    for (k, v) in dic.items():
        fwrite.write(str(k) + '\t' + str(v) + '\n')
    fwrite.close()


def read_dicts(path):
    dic = {}
    fread = open(path, 'r')
    for line in fread:
        line = line.strip().split('\t')
        dic[line[0]] = line[1]
    fread.close()
    #     print len(dic)
    return dic


def write_edges(edges, path):
    fwrite = open(path, 'w')
    for (s, d) in sorted(edges):
        fwrite.write(str(s) + '\t' + str(d) + '\n')
    fwrite.close()


def write_lists(lists, path):
    fwrite = open(path, 'w')
    for l in lists:
        fwrite.write(' '.join([str(x) for x in l]) + '\n')
    fwrite.close()


def write_labels(list, path):
    fwrite = open(path, 'w')
    for l in list:
        fwrite.write(l + '\n')
    fwrite.close()


if __name__ == '__main__':

    data = DBLP('../data/dblp-2017-04-04.xml', nodedictpath = '../model/dblp/dblp_node_dict.txt')
