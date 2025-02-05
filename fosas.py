import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, label, generate_binary_structure
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import label as ndi_label, binary_dilation
from utils.Perspectiver import Perspectiver
import concurrent.futures
def move(initial_pos, direction): return tuple(a + b for a, b in zip(initial_pos, direction))

def _get_area_groups(walker, positions:set, directions, record: set):
    next_moves = set()
    for dir in directions:
        next_pos = move(walker,directions[dir])
        if ( (next_pos in positions) and (next_pos not in (record)) ) :
            record.add(next_pos)
            next_moves.add(next_pos)
    if len(next_moves) == 0:
        record.add(walker)
    for next in next_moves:
        #print(next)
        walker = next
        record = record | _get_area_groups(walker, positions, directions, record)
    return record

def get_pits (cluster, maxcol: int, maxrow: int, previus_pits, previus_hills):
    pits = previus_pits
    hills = previus_hills
    directions={"right": (1,0), "left": (-1,0), "down":(0,1), "up":(0,-1)}
    #print(f"Tamaño: {len(cluster)}")
    while len(cluster) != 0:

        walker = cluster[0]
        #print(f"walker: {(walker)}")
        group  = _get_area_groups(walker=walker, positions=cluster, directions=directions, record=set())
        isValid = True

        for coordinate in group:
            if ( (0 in coordinate) or 
                 ( (maxrow-1) == np.array(coordinate)[0] ) or
                 ( (maxcol-1) == np.array(coordinate)[1] ) ): #Border cluster
                isValid=False
                break
            else: #Hay que comprobar que no es vecino fosas anteriormente reconocidas
                if previus_pits is not None:
                    for prev_pit in previus_pits:
                        for dir in directions:
                            if move(coordinate, directions[dir]) in prev_pit: 
                                isValid=False
                                break
                elif previus_hills is not None:
                    for prev_hill in previus_hills:
                        for dir in directions:
                            if move(coordinate, directions[dir]) in prev_hill: 
                                isValid=False
                                break

        #print(f"Tamaño antes de quitar un area: {len(cluster)}")
        cluster = [x for x in cluster if x not in group]

        #print(f"Tamaño despues: {len(cluster)}")
        if isValid: pits.append(group)
        else: hills.append(group)
        #print(pits)
    return pits, hills
            
def get_mean_pit_area(image):
    pixels = np.uint8(np.round(Perspectiver.rgb_to_grayscale(Perspectiver.normalize_to_uint8(image))))
    #pixels = image
    clusters = np.sort(np.unique(pixels))
    #print(f"Formato de la imagen: {image.shape}, clusters : {clusters}")

    pits = []
    hills = []

    MAXROW, MAXCOL = pixels.shape
    for cluster in clusters:

        #print(f"Altura del cluster: {cluster}")
        
        rows, cols = np.where(pixels == cluster)

        positions = list(zip(rows, cols))

        pits, hills = get_pits(positions, maxcol=MAXCOL, maxrow=MAXROW, previus_pits=pits, previus_hills=hills)

    area = sum([len(pit) for pit in pits])
    return area/len(pits)

def optimized_get_mean_pit_area(image):
    """
    Calcula el área media de fosas (pits) en la imagen, descartando aquellas que toquen
    el borde o estén adyacentes a fosas/colinas previas.
    """
    # Convierte la imagen a numpy y escala a grises uint8
    pixels = np.uint8(np.round(Perspectiver.rgb_to_grayscale(Perspectiver.normalize_to_uint8(image))))
    MAXROW, MAXCOL = pixels.shape

    pit_mask = np.zeros_like(pixels, dtype=bool)  # Máscara para fosas
    hill_mask = np.zeros_like(pixels, dtype=bool)  # Máscara para colinas
    pit_areas = []

    # Estructura 4-conexa
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)

    # Procesa cada intensidad en orden ascendente
    for intensity in np.sort(np.unique(pixels)):
        mask = (pixels == intensity)  # Máscara binaria para la intensidad actual
        # Etiqueta componentes conexas usando ndi_label
        labeled, num_features = ndi_label(mask, structure=structure)

        for comp in range(1, num_features + 1):
            comp_mask = (labeled == comp)
            coords = np.argwhere(comp_mask)
            # Si la componente toca el borde, se clasifica como colina
            if (np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == MAXROW - 1) or
                np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == MAXCOL - 1)):
                hill_mask |= comp_mask
                continue

            # Obtiene vecinos 4-conexos mediante dilatación
            dilated = binary_dilation(comp_mask, structure=structure)
            border = dilated & ~comp_mask
            # Si es adyacente a fosas o colinas previas, clasifica como colina
            if np.any(pit_mask & border) or np.any(hill_mask & border):
                hill_mask |= comp_mask
            else:
                pit_mask |= comp_mask
                pit_areas.append(coords.shape[0])
    
    return np.mean(pit_areas) if pit_areas else -1

def run_single_test(test_id, H, W, intensities):
    # Establece una semilla única para cada test
    np.random.seed(test_id + 1000)
    base = np.random.choice(intensities, size=(H, W))
    # Genera imagen RGB (R=G=B)
    image = np.stack([base, base, base], axis=-1)
    orig = get_mean_pit_area(image)
    opt = optimized_get_mean_pit_area(image)
    err = np.abs(orig - opt)
    return test_id, orig, opt, err

# Script de pruebas en paralelo
def test_functions(num_tests=10):
    H, W = 100, 100
    intensities = [0, 85, 170, 255, 42, 64, 123]
    total_err = 0
    error_list = []  # Almacena los errores de cada test
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_test, i+1, H, W, intensities) for i in range(num_tests)]
        for future in concurrent.futures.as_completed(futures):
            test_id, orig, opt, err = future.result()
            total_err += err
            error_list.append(err)
            if np.isclose(orig, opt):
                print(f"Test {test_id}: OK, resultado = {orig}")
            else:
                print(f"Test {test_id}: Mismatch, original = {orig}, optimizado = {opt}")
    mean_err = total_err / num_tests
    std_err = np.std(error_list)
    
    # Para calcular el error más común, redondeamos a 4 decimales
    rounded_errors = [round(e, 4) for e in error_list]
    most_common_error = Counter(rounded_errors).most_common(1)[0][0]
    
    print(f"mean_err = {mean_err}")
    print(f"std_err = {std_err}")
    print(f"error mas comun = {most_common_error}")
    
if __name__ == "__main__":
    test_functions(1000)