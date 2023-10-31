from openqaoa.problems import MaximumCut
import oq_runtime_qaoa as oq_run

from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime.program import UserMessenger

backend = IBMProvider().get_backend("ibmq_qasm_simulator")
# backend = Aer.get_backend("qasm_simulator")
user_messenger = UserMessenger()
# serialized_inputs = json.dumps(inputs, cls=RuntimeEncoder)
# deserialized_inputs = json.loads(serialized_inputs, cls=RuntimeDecoder)

maxcut_qubo = MaximumCut.random_instance(n_nodes=6, edge_probability=0.9).qubo
maxcut_qubo_dict = maxcut_qubo.asdict()

result = oq_run.main(
    backend,
    user_messenger,
    qubo_dict=maxcut_qubo_dict,
    p=1,
    n_shots=1000,
    circuit_optimization_level=1,
    optimizer_dict={"maxiter": 5, "method": "COBYLA"},
)
print(result)
