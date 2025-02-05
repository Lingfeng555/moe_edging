import numpy as np

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
            
def get_pits_fosas(image):
    #pixels = np.uint8(np.round(Perspectiver.rgb_to_grayscale(Perspectiver.normalize_to_uint8(image))))
    pixels = image
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
           
kmeans_image = np.array((
    [255, 244, 255, 255, 255],
    [244, 0  , 0  , 244, 1  ],
    [222, 0  , 3  , 1  , 3  ],
    [120, 1  , 9  , 11 , 0  ]
))
print(get_pits_fosas(kmeans_image))