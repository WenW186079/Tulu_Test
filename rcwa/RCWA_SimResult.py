import numpy as np
import os
import tensorflow as tf
from metabox import rcwa, utils, modeling
import pandas as pd
import yaml
import argparse
import inspect

gpus = tf.config.list_physical_devices('GPU')
print("\n==============Checking GPU==============\n", gpus)

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(os.getenv('TF_GPU_ALLOCATOR'))
                
def simulate_rcwa_unit_cell(
    #path
    save_model: bool = True,
    save_dir: str = './rcwa_dataset/',
    file_name: str = 'samples',
    csv_file_name: str = 'samples',
    
    #atom
    atom_material_name: str = 'Si3N4',
    pillar_range: tuple =  (0, 350e-9),
    atom_xy_pos: tuple = (0, 0),
    rotation_deg: int = 0, #rotation_deg: the rotation of the rectangle in degrees. Default: 0
    patterned_layer_thickness: float = 800e-9,
    use_linear_width: bool = True,
    num_pillar_width: int = 10, #the number of samples to take between min and max. If None,the sampling is undefined
    atom_shape: str = 'square',

    #base
    substrate_material_name: str = 'quartz',
    substrate_thickness: float = 1000e-9,
    periodicity: tuple = (350e-9, 350e-9), # grid resolution per periodicity
    
    #wavelength
    wavelength_range: tuple = (460e-9, 700e-9),
    num_wavelengths: int = 10,
    wavelength_theta: tuple = (0,), #tuple of the angles of incidence in degrees on the xz plane.
    wavelength_phi: tuple = (0,), #tuple of the angles of incidence in degrees on the yz plane
    wavelength_jones_vector:tuple = (1, 0), #Defaults to (1, 0),a linearly polarized light with the electric field vector parallel to the x axis.

    # rcwa
    harmonics: tuple = (7, 7), # Fourier orders in x and y
    resolution: int = 256,
    minibatch_size: int = 10, # number of simulations to run in parallel
    overwrite: bool = True,
    print_result: bool = True,
    return_tensor: bool = False, # tensor instead or sim_lib.simulation_output
    save_to_csv: bool = False,
):
 
    os.makedirs(save_dir, exist_ok=True)

    # Materials
    atom_material = rcwa.Material(atom_material_name)
    substrate_material = rcwa.Material(substrate_material_name)

    # Wavelengths
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_wavelengths)
    incidence = utils.Incidence(
        wavelength=wavelengths,
        theta=wavelength_theta,
        phi=wavelength_phi,
        jones_vector=wavelength_jones_vector,
        )
    print("\nincidence:\n",incidence)

    # RCWA simulation config
    sim_config = rcwa.SimConfig(
        xy_harmonics=harmonics, 
        resolution=resolution,
        minibatch_size=minibatch_size, 
        return_tensor=return_tensor,  
        return_zeroth_order= None,
        use_transmission= None,
        include_z_comp = None,
    )

    # Parameterized pillar feature
    width = utils.Feature(
        vmin=pillar_range[0], 
        vmax=pillar_range[1], 
        name="radius", 
        sampling=num_pillar_width
        )

    # Create meta-atom shape
    square = rcwa.Rectangle(
        material=atom_material, 
        x_width=width, 
        y_width=width,
        x_pos=atom_xy_pos[0],
        y_pos=atom_xy_pos[1],
        rotation_deg=rotation_deg
        )
    
    circle = rcwa.Circle(
        material=atom_material,
        radius=width,
        x_pos=atom_xy_pos[0],
        y_pos=atom_xy_pos[1],
    )

    if atom_shape == 'square':
        print('Atom shape is square')
        shapes=[square]
    elif atom_shape == 'circle':
        print('Atom shape is circle')
        shapes=[circle]
        
    # Define layers
    patterned_layer = rcwa.Layer(material=1, thickness=patterned_layer_thickness, shapes=shapes)
    substrate_layer = rcwa.Layer(material=substrate_material, thickness=substrate_thickness)

    # Assemble unit cell
    unit_cell = rcwa.UnitCell(
        layers=[patterned_layer, substrate_layer],
        periodicity=periodicity,
        refl_index = 1.0,   # the ref. index in the reflection region.
        tran_index = 1.0,   # the ref. index in the transmission region.
    )
    protocell = rcwa.ProtoUnitCell(unit_cell)

    if use_linear_width:
        custom_radius_values = np.linspace(pillar_range[0], pillar_range[1], num_pillar_width)
        parameter_tensor = tf.convert_to_tensor([custom_radius_values], dtype=tf.float32)
        print("Using use_linear_width----------------------")
    else:
        parameter_tensor = protocell.generate_initial_variables(num_pillar_width)
        print("Using random width-----------------------")

    unit_cells = protocell.generate_cells_from_parameter_tensor(parameter_tensor)

    sim_instance = rcwa.SimInstance(
        unit_cell_array=unit_cells,
        incidence=incidence,
        sim_config=sim_config
    )

    sim_result = rcwa.simulate(sim_instance)

    if print_result:
        print_full_sim_result(sim_result, wavelengths)

    sim_lib = modeling.SimulationLibrary(
        protocell=protocell,
        incidence=incidence,
        sim_config=sim_config,
        feature_values=parameter_tensor.numpy(),    
        simulation_output=sim_result                
    )

    if save_model:
        modeling.save_simulation_library(sim_lib, name=file_name, path=save_dir, overwrite=overwrite)
        print(f"RCWA simulation saved to {os.path.join(save_dir, file_name)}.pkl")

    if save_to_csv:
        save_path = os.path.join(save_dir, f"{csv_file_name}.csv")
        save_csv_file(sim_result, save_path, sim_lib)

    return sim_lib

def print_full_sim_result(sim_result: rcwa.SimResult, wavelengths: np.ndarray):
    print("====================Simulation output for first sample:==================\n")

    def print_tensor(name, tensor):
        try:
            val = tensor[0, 0, ...]  # the frist wavelength，the first sample
            print(f"{name}: shape={val.shape}")
            print(val.numpy())
        except Exception as e:
            print(f"{name} printing failed: {e}")

    for attr_name in dir(sim_result):
        if attr_name.startswith("_"):
            continue
        try:
            tensor = getattr(sim_result, attr_name)
            if hasattr(tensor, "shape") and hasattr(tensor, "__getitem__"):
                print_tensor(attr_name, tensor)
        except Exception as e:
            print(f"Error accessing attribute {attr_name}: {e}")
    
    # rx: the x component of the reflected diffraction coeff. 
    # ry: the y component of the reflected diffraction coeff.
    # rz: the z component of the reflected diffraction coeff.
    # r_eff: the reflective efficiency. 			
    # r_power: the total reflected power.
    # tx: the x component of the transmitted diffraction coeff. 	
    # ty: the y component of the transmitted diffraction coeff.	
    # tz: the z component of the transmitted diffraction coeff.	
    # t_eff: the transmissive efficiency.			
    # t_power: the total transmitted power. 

    print("\n=================== Done printing=========================")

def save_csv_file(sim_result, save_path="rcwa.csv", sim_lib=None):
    t_power = getattr(sim_result, "t_power", None)
    r_power = getattr(sim_result, "r_power", None)

    if t_power is None or r_power is None or sim_lib is None:
        print("Missing data (t_power, r_power, or sim_lib)")
        return

    wavelengths = np.array(sim_lib.incidence.wavelength)  # (num_wavelengths,)
    feature_values = np.array(sim_lib.feature_values).squeeze()  # (num_samples,)
    xy_harmonics = sim_lib.sim_config.xy_harmonics

    t_power = t_power.numpy().squeeze() # shape: (wavelengths, samples)
    r_power = r_power.numpy().squeeze()

    rows = []
    if wavelengths.size == 1 and np.ndim(feature_values) == 0:
        rows.append({
            "wavelength (m)": float(wavelengths),
            "pillar width (m)": float(feature_values),
            "t_power": float(t_power),
            "r_power": float(r_power),
            "x_harmonics": xy_harmonics[0],
            "y_harmonics": xy_harmonics[1],
        })
    elif wavelengths.size == 1 and np.ndim(feature_values) == 1:
        for j, width in enumerate(feature_values):
            rows.append({
                "wavelength (m)": float(wavelengths),
                "pillar width (m)": width,
                "t_power": float(t_power[j]),
                "r_power": float(r_power[j]),
                "x_harmonics": xy_harmonics[0],
                "y_harmonics": xy_harmonics[1],
            })
    elif wavelengths.size > 1 and np.ndim(feature_values) == 0:
        for i, wl in enumerate(wavelengths):
            rows.append({
                "wavelength (m)": wl,
                "pillar width (m)": float(feature_values),
                "t_power": float(t_power[i]),
                "r_power": float(r_power[i]),
                "x_harmonics": xy_harmonics[0],
                "y_harmonics": xy_harmonics[1],
            })
    else:
        for i, wl in enumerate(wavelengths):
            for j, width in enumerate(feature_values):
                rows.append({
                    "wavelength (m)": wl,
                    "pillar width (m)": width,
                    "t_power": t_power[i, j],
                    "r_power": r_power[i, j],
                    "x_harmonics": xy_harmonics[0],
                    "y_harmonics": xy_harmonics[1],
                })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Saved detailed t_power and r_power to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    sim_signature = inspect.signature(simulate_rcwa_unit_cell)
    sim_config = {k: v for k, v in config.items() if k in sim_signature.parameters}

    simulate_rcwa_unit_cell(**sim_config)

   
if __name__ == "__main__":
    main()

 
 