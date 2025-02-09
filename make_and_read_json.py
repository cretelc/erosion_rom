import json
import pandas as pd

class InputParameter:
    def __init__(self, value:float, description:str,  unit:str):
        self.description = description
        self.value = value
        self.unit = unit
        
    def get_dict(self):   
        return {'value': self.value, 'description': self.description, 'unit': self.unit}


def make_json():
    # Discharge
    # Grids 
    # Plume
    # Facility
    json_dict = {
        'eta_p'     : InputParameter(0.004,     "cex ion impingement probability in the plume", "").get_dict(),
        'eta_b1'    : InputParameter(0.75,      "upstream cex ion impingement probability between the grids", "").get_dict(),
        'eta_b2'    : InputParameter(None,      "downstream cex ion impingement probability between grids", "").get_dict(), 
        'n_ds'      : InputParameter(10,        "number of thruster radii downstream", "").get_dict(),
        'Rebs'      : InputParameter(0.001,     "electron backstreaming ratio limit", "").get_dict(),
        'w'         : InputParameter(0.5,       "weight of upstream accel vertice used in ra_avg calc, [0,1]", "").get_dict(),
        'lg'        : InputParameter(0.36,      "grid separation", "mm").get_dict(),
        'rs'        : InputParameter(1.91/2,    "screen grid radius", "mm").get_dict(),
        'ra'        : InputParameter(1.14/2,    "accel grid radius", "mm").get_dict(),
        's'         : InputParameter(2.24,      "aperture center spacing", "mm").get_dict(),
        'ta'        : InputParameter(0.38,      "accel grid thickness", "mm").get_dict(),
        'ts'        : InputParameter(0.38,      "screen grid thickness", "mm").get_dict(),
        'Rth'       : InputParameter(0.15,      "thruster major radius", "m").get_dict(),
        'M_grid'    : InputParameter(95.95,     "grid material atomic mass", "g/mol").get_dict(),
        'rho_grid'  : InputParameter(0.01022,   "grid material density", "g/mm3").get_dict(),
        'diverg'    : InputParameter(20,        "plume divergence half angle", 'deg').get_dict(),
        'phi_p'     : InputParameter(0,         "plasma potential in the beam", "V").get_dict(),
        'ni_dis'    : InputParameter(1e17,      "discharge ion density", "m-3").get_dict(),
        'Te_dis'    : InputParameter(15,        "discharge electron temperature", "eV").get_dict(),
        'n0_fac'    : InputParameter(2e18,      "facility neutral density", "m-3").get_dict(),
        'Rch'       : InputParameter(2,         "cylindrical vacuum chamber radius", "m").get_dict(),
        'Lch'       : InputParameter(3,         "cylindrical vacuum chamber length", "m").get_dict(),
        'Ns'        : InputParameter(6,         "vacuum chamber wall material atomic number", "").get_dict(), 
        'Ms'        : InputParameter(12.011,    "vacuum chamber wall material atomic mass", "g/mol").get_dict(),
        'rho_C'     : InputParameter(2.25e6,    "redposited material density", "g/m3").get_dict(),
        'Te_beam'   : InputParameter(2,         "beam electron temperature", "eV").get_dict(),
        'nbp'       : InputParameter(5e16,      "downstream plasma density near accel grid", "m-3").get_dict(),
        'IB'        : InputParameter(1.76,      "total beam current", "A").get_dict(),
        'Ib'        : InputParameter(2.7e-4,    "beamlet current", "A").get_dict(),
        'Vd'        : InputParameter(1100,      "total (discharge) potential", "V").get_dict(),
        'Va'        : InputParameter(-180,      "applied accel grid potential", "V").get_dict(),
        'phi_dis'   : InputParameter(32,        "discharge plasma potential over screen grid", "V").get_dict(),
        'Ni'        : InputParameter(54,        "propellant atomic number", "").get_dict(), 
        'Mi'        : InputParameter(134.27,    "propellant atomic mass", "g/mol").get_dict(),
        'Ec'        : InputParameter(400,       "charge-exchange collision energy", "eV").get_dict(), 
        'E_cex'     : InputParameter(400,       "erosion impingement energy", "eV").get_dict()
    }
    
    json_dict['eta_b2']['value'] = 1-json_dict['eta_b1']['value']

    print(json_dict)
    # Serializing json
    json_object = json.dumps(json_dict, indent=1)
 
# Writing to sample.json
    with open("rom_inputs.json", "w") as outfile:
        outfile.write(json_object)
        print(json_dict)
        return

def read_to_excel(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.transpose()
    df.to_excel('inputs.xlsx')
    return

if __name__ == "__main__":
    read_to_excel('sample.json')