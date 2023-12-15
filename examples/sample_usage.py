import perfograph as pg

cpp_graph = pg.from_file('samples/program.cpp')
c_graph = pg.from_file('samples/program.c')
ll_graph = pg.from_file('samples/program.ll')

pg.to_dot(cpp_graph, 'sample_graph.pdf')
pg.to_dot(cpp_graph, 'sample_graph.png')
pg.to_json(cpp_graph, 'sample_json_graph.json')