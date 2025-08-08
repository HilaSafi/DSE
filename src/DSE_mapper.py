import qiskit
from qiskit import transpile, IBMQ
from qiskit.circuit.quantumcircuit import QuantumCircuit
import math
import matplotlib.pyplot as plt
from qiskit import Aer, dagcircuit
from qiskit.transpiler import CouplingMap
from qiskit.providers.fake_provider import ConfigurableFakeBackend, FakeBrooklyn, FakeBackendV2
from qiskit.circuit.library import DraperQFTAdder, QuantumVolume, CDKMRippleCarryAdder, QFT
from qiskit.pulse import Schedule, InstructionScheduleMap, DriveChannel, ControlChannel, Play, Gaussian
from qiskit.circuit.random import random_circuit
import os
from openpyxl import load_workbook
from os import listdir
from os.path import isfile, join
import pandas as pd
from qiskit.visualization import plot_gate_map
from multiprocessing import Pool, Lock
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import OptimizeCliffords,Optimize1qGates, Optimize1qGatesSimpleCommutation, HoareOptimizer, Optimize1qGatesDecomposition, RemoveDiagonalGatesBeforeMeasure, CommutativeCancellation, CommutativeInverseCancellation, CXCancellation, CrosstalkAdaptiveSchedule
import topology_functions as tf

def setup1(pass_manager):
    pass_manager.append(Optimize1qGates())
    pass_manager.append(OptimizeCliffords())
    pass_manager.append(Optimize1qGatesSimpleCommutation())

def setup2(pass_manager):
    pass_manager.append(Optimize1qGatesDecomposition())
    pass_manager.append(CXCancellation())

def setup3(pass_manager):
    setup1(pass_manager)
    pass_manager.append(RemoveDiagonalGatesBeforeMeasure())

def setup4(pass_manager):
    pass_manager.append(Optimize1qGatesDecomposition())
    pass_manager.append(CommutativeCancellation())

def setup5(pass_manager):
    setup3(pass_manager)
    try:
        pass_manager.append(HoareOptimizer())
    except Exception as e:
        print(e)
        return
    pass_manager.append(CommutativeCancellation())
    pass_manager.append(CommutativeInverseCancellation())

#Will be used for simulations with noise
def compiler_with_noise():
    optimization_level = 3
    routing_method = 'sabre'
    layout_method = 'sabre'
    scheduling_method = 'alap'
    pass_manager = PassManager()
    pass_manager.append(OptimizeCliffords())
    pass_manager.append(CrosstalkAdaptiveSchedule())
    pass_manager.append(RemoveDiagonalGatesBeforeMeasure())
    pass_manager.append(Optimize1qGatesSimpleCommutation())
    pass_manager.append(HoareOptimizer())
    pass_manager.append(CommutativeInverseCancellation())


def process_benchmark(b):
    #data before compilation
    circuit = QuantumCircuit.from_qasm_file(b[0])
    depth_before = circuit.depth()
    gates_before = circuit.size()
    gate_types_before = circuit.count_ops()
    #b[1] = b[1].replace('.qasm','')
    twoqgates_before = 0
    twoqgates_before = circuit.count_ops().get('cz', 0) + circuit.count_ops().get('cx', 0)
    l_data = []

    for dcm in devices_coupling_maps:

        num_phys_qubits = dcm[2]

        if circuit.num_qubits <= num_phys_qubits:
            # Transpile the circuit for the target device
            for ol in optimization_levels:
                for rt in routing_techniques:
                    #for sm in scheduling_methods:
                        for lm in layout_methods:
                            for v in range(6):
                                pm = PassManager()
                                if v>0:
                                    if v==5:
                                        if b[1] not in ['20QBT_100CYC_QSE_8.qasm','cycle10_2_110.qasm','plus63mod4096_163.qasm','q=6_s=19994_2qbf=05_1.qasm','q=8_s=39992_2qbf=08_1.qasm','shor_15.qasm','shor_35.qasm','square_root_7.qasm']:
                                            setup5(pm) 
                                        else: 
                                            pm.append(CommutativeCancellation())
                                            pm.append(CommutativeInverseCancellation())
                                    variable_setup[v-1](pm)
                                try:
                                  pm.run(circuit)
                                  transpiled_circuit = transpile(circuit, basis_gates=['x','y','z','rx','ry','rz','cx','cy'],coupling_map = dcm[1],optimization_level = ol,layout_method=lm, routing_method = rt #scheduling_method = sm
                                )
                                except Exception as e:
                                    print(e)
                                    return

                                # Get the number of swaps, depth and fidelity after compilation
                                swaps = transpiled_circuit.count_ops().get('swap', 0)
                                depth = transpiled_circuit.depth()
                                gates = transpiled_circuit.size()
                                gate_types = transpiled_circuit.count_ops()
                                twoqgates = transpiled_circuit.count_ops().get('cz', 0) + transpiled_circuit.count_ops().get('cx', 0) + 3*swaps
                                

                                # Save the data
                                d = [b[1].replace('.qasm',''), dcm[0], ol, rt,lm,v, gates_before, gates, gate_types_before,gate_types ,twoqgates_before,twoqgates,swaps, depth_before,depth]
                                l_data.append(d)
    print(b[1])
    return l_data

if __name__ == '__main__':

    curdir = os.path.dirname(__file__)


    # Define your quantum circuit
    benchmarks_dir = os.path.join(curdir,"circuits/")
    benchmarks = [f for f in listdir(benchmarks_dir) if join(benchmarks_dir, f)] 

    # # Choose the target device
    
    devices_coupling_maps = [("Google Bristlecone",[[0,1],  [1,2],  [2,3],  [3,4],  [4,5], [5,6],  [6,7], [7,8],  [8,9], [9,10],  [10,11], [12,13], [13,14],  [14,15],  [15,16], [16,17], [17,18], [18,19],  [19,20], [20,21],  [21,22],  [22,23], [24,25],  [25,26],  [26,27], [27,28],[28,29], [29,30], [31,32],  [32,33],  [33,34],  [34,35],[36,37], [37,38],  [38,39],  [39,40], [40,41], [42,43],  [43,44],  [44,45], [45,46],  [46,47], [48,49],[49,50],[50,51], [51,52],  [52,53], [53,54], [54,55],  [55,56],  [56,57], [57,58], [58,59],[60,61],  [61,62], [62,63],  [63,64], [64,65],  [65,66],  [66,67], [67,68],  [68,69],  [69,70],  [70,71],[1,12],[1,14],[14,3],[3,16],[16,5],[5,18],[18,7],[7,20],[9,20],[9,22],[22,11],[13,24],[13,26],[15,26],[15,28],[28,17],[30,17],[30,19],[19,32],[32,21],[21,34],[34,23],[25,36],[25,38],[38,27],[40,27],[40,29],[42,29],[42,31],[31,44],[44,33],[33,46],[46,35],[37,48],[37,50],[50,39],[39,52],[52,41],[41,54],[54,43],[43,56],[45,56],[45,58],[58,47],[49,60],[49,62],[62,51],[51,64],[64,53],[53,66],[66,55],[55,68],[68,57],[57,70],[70,59], [1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 9], [11, 10], 
                            [13, 12], [14, 13], [15, 14], [16, 15], [17, 16], [18, 17], [19, 18], [20, 19], [21, 20], [22, 21], [23, 22],[25, 24], [26, 25], [27, 26], [28, 27], [29, 28], [30, 29], [32, 31], [33, 32], [34, 33], [35, 34], [37, 36], [38, 37], [39, 38], [40, 39], [41, 40], [43, 42], [44, 43], [45, 44], [46, 45], [47, 46],[49, 48], [50, 49], [51, 50], [52, 51], [53, 52], [54, 53], [55, 54], [56, 55], [57, 56], [58, 57], [59, 58], [61, 60], [62, 61], [63, 62], [64, 63], [65, 64], [66, 65], [67, 66], [68, 67], [69, 68], [70, 69], [71, 70], [12, 1], [14, 1], [3, 14], [16, 3], [5, 16], [18, 5], [7, 18], [20, 7], [20, 9], [22, 9], [11, 22], [24, 13], [26, 13], [26, 15], [28, 15], [17, 28], [17, 30], [19, 30], [32, 19], [21, 32], [34, 21], [23, 34],       [36, 25], [38, 25], [27, 38], [27, 40], [29, 40], [29, 42], [31, 42], [44, 31], [33, 44], [46, 33], [35, 46],[48, 37], [50, 37], [39, 50], [52, 39], [41, 52], [54, 41], [43, 54], [56, 43], [56, 45], [58, 45], [47, 58],[60, 49], [62, 49], [51, 62], [64, 51], [53, 64], [66, 53], [55, 66], [68, 55], [57, 68], [70, 57], [59, 70]],72)]
    
    #add random topologies
    connectivity_density=[0.013895, 0.03, 0.05,  0.1, 0.3, 0.5, 0.8]
    backend = ConfigurableFakeBackend(name='StartingBackend',n_qubits=128, coupling_map=[
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], 
    [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 28], 
    [28, 29], [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], 
    [41, 42], [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], 
    [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 60], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], 
    [67, 68], [68, 69], [69, 70], [70, 71], [71, 72], [72, 73], [73, 74], [74, 75], [75, 76], [76, 77], [77, 78], [78, 79], [79, 80], 
    [80, 81], [81, 82], [82, 83], [83, 84], [84, 85], [85, 86], [86, 87], [87, 88], [88, 89], [89, 90], [90, 91], [91, 92], [92, 93], 
    [93, 94], [94, 95], [95, 96], [96, 97], [97, 98], [98, 99], [99, 100], [100, 101], [101, 102], [102, 103], [103, 104], [104, 105], 
    [105, 106], [106, 107], [107, 108], [108, 109], [109, 110], [110, 111], [111, 112], [112, 113], [113, 114], [114, 115], [115, 116], 
    [116, 117], [117, 118], [118, 119], [119, 120], [120, 121], [121, 122], [122, 123], [123, 124], [124, 125], [125, 126], [126, 127],[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 9], [11, 10], [12, 11], [13, 12], [14, 13], [15, 14], 
    [16, 15], [17, 16], [18, 17], [19, 18], [20, 19], [21, 20], [22, 21], [23, 22], [24, 23], [25, 24], [26, 25], [27, 26], [28, 27], 
    [29, 28], [30, 29], [31, 30], [32, 31], [33, 32], [34, 33], [35, 34], [36, 35], [37, 36], [38, 37], [39, 38], [40, 39], [41, 40], 
    [42, 41], [43, 42], [44, 43], [45, 44], [46, 45], [47, 46], [48, 47], [49, 48], [50, 49], [51, 50], [52, 51], [53, 52], [54, 53], 
    [55, 54], [56, 55], [57, 56], [58, 57], [59, 58], [60, 59], [61, 60], [62, 61], [63, 62], [64, 63], [65, 64], [66, 65], [67, 66], 
    [68, 67], [69, 68], [70, 69], [71, 70], [72, 71], [73, 72], [74, 73], [75, 74], [76, 75], [77, 76], [78, 77], [79, 78], [80, 79], 
    [81, 80], [82, 81], [83, 82], [84, 83], [85, 84], [86, 85], [87, 86], [88, 87], [89, 88], [90, 89], [91, 90], [92, 91], [93, 92], 
    [94, 93], [95, 94], [96, 95], [97, 96], [98, 97], [99, 98], [100, 99], [101, 100], [102, 101], [103, 102], [104, 103], [105, 104], 
    [106, 105], [107, 106], [108, 107], [109, 108], [110, 109], [111, 110], [112, 111], [113, 112], [114, 113], [115, 114], [116, 115], 
    [117, 116], [118, 117], [119, 118], [120, 119], [121, 120], [122, 121], [123, 122], [124, 123], [125, 124], [126, 125], [127, 126]])
    for d in connectivity_density:
       devices_coupling_maps.append(('Custom_' + str(d),tf.increase_coupling_density(backend.coupling_map, d),128))


        
    data = []

    #list of options
    scheduling_methods = ['asap','alap']
    optimization_levels = [0,1,2] 
    layout_methods = ['trivial','dense','sabre']
    routing_techniques = ['stochastic', 'sabre']
    variable_setup = [setup1, setup2, setup3, setup4, setup5]
    CrosstalkAdaptiveSchedule = [True,False]



    paths = [(benchmarks_dir + b,b) for b in benchmarks]
    with Pool(10) as p:
        results = p.map(process_benchmark, paths)

    for res in results:
        data.extend(res)
        
    #[processBenchmark(p) for p in paths]
    df =  pd.DataFrame(data, columns =   ["benchmarks names", "device name", "optimization level","routing technique","layout method","setup number","gates before", "num of gates", "different gates count before", "different gates count", "2qgates before","2qgates","number of swaps","depth before","depth"]) 
    df.to_excel(curdir + "DSE_results.xlsx", index=False)




