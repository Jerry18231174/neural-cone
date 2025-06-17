import xml.etree.ElementTree as ET
import json
import argparse
import re


def parse_element(xml, json, variables):
    bsdf_count = 0
    shape_count = 0
    for child in xml:
        if child.tag == "bsdf":
            bid: str = child.attrib["id"] if "id" in child.attrib else "BSDF_{:02d}".format(bsdf_count)
            btype: str = child.attrib["type"]
            bsdf = {"type": btype}
            if "bsdf" not in json:
                json["bsdf"] = {}
            json["bsdf"][bid] = bsdf
            parse_element(child, bsdf, variables)
            bsdf_count += 1
        elif child.tag == "shape":
            sid: str = child.attrib["id"] if "id" in child.attrib else "Shape_{:02d}".format(shape_count)
            stype: str = child.attrib["type"]
            shape = {"type": stype}
            json["shape"][sid] = shape
            parse_element(child, shape, variables)
            shape_count += 1
        elif child.tag in ["sensor", "emitter", "integrator"]:
            mtype: str = child.attrib["type"]
            module = {}
            if mtype.startswith("$"):
                module["type"] = json[mtype[1:]]
            else:
                module["type"] = mtype
            json[child.tag] = module
            parse_element(child, module, variables)
        # Name-type pairs
        elif child.tag in ["texture"]:
            name: str = child.attrib["name"]
            mtype: str = child.attrib["type"]
            module = {"type": mtype}
            json[name] = module
            parse_element(child, module, variables)
        # Name-value pairs (not-recursive)
        elif child.tag in ["float", "integer", "string", "boolean", "rgb",
                           "default"]:
            name: str = child.attrib["name"]
            value: str = child.attrib["value"]
            if value.startswith("$") and value[1:] in variables:
                value = variables[value[1:]]

            if child.tag == "float":
                json[name] = float(value)
            elif child.tag == "integer":
                json[name] = int(value)
            elif child.tag == "string":
                json[name] = value
            elif child.tag == "boolean":
                json[name] = value.lower() == "true"
            elif child.tag == "rgb":
                json[name] = [float(v) for v in re.split(r"[,\s]+", value)]
            elif child.tag == "default":
                json[name] = int(value) if value.isdigit() else value
            variables[name] = json[name]
        # Name-only elements
        elif child.tag in ["transform"]:
            name: str = child.attrib["name"]
            module = {"name": name}
            if child.tag == "transform":
                json["transform"] = module
            parse_element(child, module, variables)
        # Type-only elements
        elif child.tag in ["film", "rfilter", "sampler"]:
            mtype: str = child.attrib["type"]
            module = {"type": mtype}
            json[child.tag] = module
            parse_element(child, module, variables)
        # Value-only elements
        elif child.tag in ["matrix"]:
            value: str = child.attrib["value"]
            if value.startswith("$") and value[1:] in variables:
                value = variables[value[1:]]
            
            if child.tag == "matrix":
                json["matrix"] = [float(v) for v in re.split(r"[,\s]+", value)]
        # Id-only elements
        elif child.tag == "ref":
            id: str = child.attrib["id"]
            json["ref"] = id
        elif child.tag == "scale":
            json["scale"] = (float(child.attrib["x"]), float(child.attrib["y"]))
        else:
            raise ValueError(f"Unknown XML tag: {child.tag}")



def convert(xml_path, json_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    node = {
        "version": "0.1.0",
        "integrator": None,
        "sensor": None,
        "bsdf": {},
        "shape": {},
    }
    variables = {}

    # Convert the XML to a dictionary
    parse_element(root, node, variables)

    # Write the dictionary to a JSON file
    with open(json_path, 'w') as json_file:
        json.dump(node, json_file, indent=4)
    
    print(f"Converted {xml_path} to {json_path} successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XML to JSON.")
    parser.add_argument("xml_path", type=str, help="Path to the input XML file.")
    parser.add_argument("json_path", type=str, help="Path to the output JSON file.")
    
    args = parser.parse_args()
    
    convert(args.xml_path, args.json_path)