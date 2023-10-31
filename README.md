# openqaoa-qiskit-lite-runtime
A lite version of openqaoa-qiskit that supports execution of a QAOA workflow natively on IBM servers as a runtime program


## Setting up the runtime program

Since this is a custom runtime program, we first need to upload the program onto IBM servers before we can start using the program to execute QAOA workflows. The script `runtime_upload.py` lets users do exactly that. Run this script with your IBM account credentials to upload the custom runtime script to your account.

Once the program is uploaded successfully, you will receive a unique `program_id` that identifies your custom runtime program. To run QAOA workflows using this runtime program, you can specify this `program_id` to access the program.
