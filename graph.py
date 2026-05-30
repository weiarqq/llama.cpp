import json



with open("qwen3.5-moe-graph-by-layer.json") as f:
    graph = json.load(f)
print("global_after_nodes")
print(len(graph['graphs'][1]['global_after']['nodes']))

for node in graph['graphs'][1]['global_after']['nodes']:
    print(node['name'])

print("global_before_nodes")
print(len(graph['graphs'][1]['global_before']['nodes']))

for node in graph['graphs'][1]['global_before']['nodes']:
    print(node['name'])

print("---"*10)
print("global_layers_nodes")
print(len(graph['graphs'][1]['layers']))
print("---"*10)
for node in graph['graphs'][1]['layers'][4:5]:
    layer_index = node['layer']
    nodes = node['nodes']
    fnodes = []
    for node in nodes:
        op = node['op']
        name = node['name']
        if op == "RESHAPE" or op == "TRANSPOSE" or op == "VIEW" or op == "PERMUTE" or op == "NONE":
            continue
        if name.startswith("node_"):
            continue
        if name.startswith("cache"):
            continue
        if name.startswith("norm"):
            continue
        print(node['name'], op)
        fnodes.append(node)
    print(f"layer {layer_index} : {len(fnodes)}")


