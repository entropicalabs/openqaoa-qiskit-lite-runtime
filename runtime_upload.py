# script to upload runtime program on IBM account
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

oq_runtime_data = "openqaoa_qiskit_lite_source_code.py"
oq_runtime_json = "openqaoa_qiskit_lite_source_code_metadata.json"

program_id = service.upload_program(data=oq_runtime_data, metadata=oq_runtime_json)
print(program_id)
