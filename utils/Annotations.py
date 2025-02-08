import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd

class XMLFolderParser:
    """Clase para parsear archivos XML y convertirlos en un DataFrame."""

    def __init__(self, folder_path):
        """
        Inicializa la clase con la ruta de la carpeta.
        :param folder_path: Ruta a la carpeta con archivos XML.
        """
        self.folder_path = folder_path

    def parse_xml_file(self, xml_file):
        """
        Parsea un archivo XML y extrae la información.
        :param xml_file: Ruta al archivo XML.
        :return: Lista de diccionarios con datos extraídos.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Datos generales del XML
        folder = root.find('folder').text if root.find('folder') is not None else None
        filename = root.find('filename').text if root.find('filename') is not None else None
        database = root.find('source/database').text if root.find('source/database') is not None else None
        size = root.find('size')
        width = int(size.find('width').text) if size.find('width') is not None else None
        height = int(size.find('height').text) if size.find('height') is not None else None
        depth = int(size.find('depth').text) if size.find('depth') is not None else None

        data_list = []
        # Itera sobre cada objeto presente en el XML
        for obj in root.findall('object'):
            obj_name = obj.find('name').text if obj.find('name') is not None else None
            pose = obj.find('pose').text if obj.find('pose') is not None else None
            truncated = int(obj.find('truncated').text) if obj.find('truncated') is not None else None
            difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else None
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text) if bndbox.find('xmin') is not None else None
            ymin = int(bndbox.find('ymin').text) if bndbox.find('ymin') is not None else None
            xmax = int(bndbox.find('xmax').text) if bndbox.find('xmax') is not None else None
            ymax = int(bndbox.find('ymax').text) if bndbox.find('ymax') is not None else None

            # Compila la información en un diccionario
            data = {
                'folder': folder,
                'filename': filename,
                'database': database,
                'width': width,
                'height': height,
                'depth': depth,
                'object_name': obj_name,
                'pose': pose,
                'truncated': truncated,
                'difficult': difficult,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            }
            data_list.append(data)
        return data_list

    def parse_folder(self):
        """
        Parsea todos los archivos XML de la carpeta y devuelve un DataFrame.
        :return: DataFrame con los datos extraídos.
        """
        all_data = []
        # Busca todos los archivos XML en la carpeta
        for xml_file in glob.glob(os.path.join(self.folder_path, '*.xml')):
            all_data.extend(self.parse_xml_file(xml_file))
        return pd.DataFrame(all_data)

# Ejemplo de uso:
if __name__ == "__main__":
    parser = XMLFolderParser('ANNOTATIONS/')
    df = parser.parse_folder()

    counts = df.groupby(['folder', 'filename']).size().reset_index(name='object_count')
    max_obj = counts.loc[counts['object_count'].idxmax()]
    print(max_obj)

