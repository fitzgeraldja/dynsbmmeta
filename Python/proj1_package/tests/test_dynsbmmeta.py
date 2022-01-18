import pytest 
from proj1_package import dynsbmmeta 

# example base test -- correct if needed, learn pytest properly in future
# once all done, can run test with pytest tests/ from terminal. Include --pdb to stop execution whenever one of them fails, and open interactive debugging session. 
# Can also open at specific line if include there "from pdb import set_trace; set_trace()", then run tests with pytest tests/ -s. Can instead start regular session at that line if use "from IPython import embed; embed()"

@pytest.mark.parametrize( 
    'name, expected', # i.e. pass representative inputs and their appropriate outputs
    [  
        ['val','expected res'],
        ...
    ],
)
def test_dynsbmmeta_inference(name,expected):
    assert dynsbmmeta.inference(name) == expected
    pass