import shutil
import os
from xml.dom.minidom import parse
from src.utils.config import get_config, to_windows_path

class QLCHandler:
    def __init__(self, filename, setup_path):
        self.showfile_path = setup_path
        self.shows = {}
        self.filename = filename

        program_data_path = get_config("program_data_path")
        directory = f'{program_data_path}/AIQLCshows/shows'
        # Check if the shows-directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create the setup directory if it doesn't exist
        directory = os.path.join(directory, self.filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.destination_folder = directory
        print(self.destination_folder)
        
    @staticmethod
    def remove_whitespace_nodes(node, unlink=False):
        """Removes all of the whitespace-only text descendants of node.
        If unlink is True, also unlinks the nodes from their parents."""
        remove_list = []
        for child in node.childNodes:
            if child.nodeType == node.TEXT_NODE and not child.data.strip():
                remove_list.append(child)
            elif child.hasChildNodes():
                QLCHandler.remove_whitespace_nodes(child, unlink)
        for node in remove_list:
            node.parentNode.removeChild(node)
            if unlink:
                node.unlink()

    def get_path(self, showname):
        path = self.shows[showname]["absolute_path"]
        return path

    def create_copy(self, showname):
        destination_folder = self.destination_folder
        absolute_path = f"{destination_folder}/{showname}.qxw"
        shutil.copy2(self.showfile_path, absolute_path)
        self.shows[showname] = {"Functions": [], "Buttons": []}
        self.shows[showname]["absolute_path"] = absolute_path
        self.shows[showname]["dom"] = parse(absolute_path)

    def add_element(self, showname, parent_element_name, element_name, attributes={}, text=None, child_elements=[]):
        # Create the new element
        dom = self.shows[showname]["dom"]
        element = dom.createElement(element_name)
        for key, value in attributes.items():
            element.setAttribute(key, value)
        if text is not None:
            element.appendChild(dom.createTextNode(text))

        # Add child elements
        for child_element in child_elements:
            element.appendChild(child_element)

        # Find the parent element and append the new element to it
        parent_element = dom.getElementsByTagName(parent_element_name)[0]
        parent_element.appendChild(element)

        # Remove whitespace nodes
        QLCHandler.remove_whitespace_nodes(dom)
        showpath = self.shows[showname]["absolute_path"]
        # Write the XML declaration, DOCTYPE declaration, and XML string to the file
        with open(showpath, 'w') as f:
            f.write(dom.toprettyxml(indent="  "))

    def add_script(self, showname, script, function_name):
        # Create the Speed, Direction, and RunOrder elements
        dom = self.shows[showname]["dom"]

        speed = dom.createElement('Speed')
        speed.setAttribute('FadeIn', '0')
        speed.setAttribute('FadeOut', '0')
        speed.setAttribute('Duration', '0')

        direction = dom.createElement('Direction')
        direction.appendChild(dom.createTextNode('Forward'))

        run_order = dom.createElement('RunOrder')
        run_order.appendChild(dom.createTextNode('Loop'))

        # Create the Command elements
        commands = []
        for command in script:
            command_elem = dom.createElement('Command')
            command_elem.appendChild(dom.createTextNode(command))
            commands.append(command_elem)

        function_info = {'ID': str(len(self.shows[showname]["Functions"])), 'Type': 'Script', 'Name': str(function_name)}

        # Add the Function element to the XML file
        self.add_element(showname, 'Engine', 'Function', function_info, None, [speed, direction, run_order] + commands)
        self.shows[showname]["Functions"].append(function_info)

        return str(len(self.shows[showname]["Functions"])-1)

    def add_track(self, show_scripts, showname, function_names):
        dom = self.shows[showname]["dom"]
        # Initialize scripts with powershell command
        script_ids = [
            self.add_script(
                showname, 
                [self.powershell_script(f"{showname}.wav")],
                "powershell_script"
            )
        ]
        # Assume add_script is a function that adds a script and returns its ID
        for i in range(len(show_scripts)):
            script_ids.append(self.add_script(showname, show_scripts[i], function_names[i]))

        attributes = {"ID": str(len(self.shows[showname]["Functions"])), "Type": "Collection", "Name": "New Collection 14"}
        self.add_element(showname, "Engine", "Function", attributes)

        # Get the newly created collection
        engine = dom.getElementsByTagName("Engine")[0]
        collection = engine.getElementsByTagName("Function")[-1]

        # Add the script IDs as steps
        for i, script_id in enumerate(script_ids):
            attributes = {"Number": str(i)}
            step = dom.createElement("Step")
            for key, value in attributes.items():
                step.setAttribute(key, value)
            step.appendChild(dom.createTextNode(str(script_id)))
            collection.appendChild(step)

        self.shows[showname]["Functions"].append(attributes)
        showpath = self.shows[showname]["absolute_path"]
        # Write the XML declaration, DOCTYPE declaration, and XML string to the file
        with open(showpath, 'w') as f:
            f.write(dom.toprettyxml(indent="  "))

    def add_chaser(self, showname, chaserid, chasername, duration=10000):
        # Create the chaser element
        dom = self.shows[showname]["dom"]
        attributes = {"ID": str(len(self.shows[showname]["Functions"])), "Type": "Chaser", "Name": chasername}
        chaser = dom.createElement("Function")
        for key, value in attributes.items():
            chaser.setAttribute(key, value)

        id = str(len(self.shows[showname]["Functions"]))

        # Create the Speed element
        speed = dom.createElement("Speed")
        speed.setAttribute("FadeIn", "0")
        speed.setAttribute("FadeOut", "0")
        speed.setAttribute("Duration", str(duration))
        chaser.appendChild(speed)

        # Create the Direction element
        direction = dom.createElement("Direction")
        direction.appendChild(dom.createTextNode("Forward"))
        chaser.appendChild(direction)

        # Create the RunOrder element
        run_order = dom.createElement("RunOrder")
        run_order.appendChild(dom.createTextNode("Loop"))
        chaser.appendChild(run_order)

        # Create the SpeedModes element
        speed_modes = dom.createElement("SpeedModes")
        speed_modes.setAttribute("FadeIn", "Default")
        speed_modes.setAttribute("FadeOut", "Default")
        speed_modes.setAttribute("Duration", "Common")
        chaser.appendChild(speed_modes)

        # Create the Step element
        step = dom.createElement("Step")
        step.setAttribute("Number", "0")
        step.setAttribute("FadeIn", "0")
        step.setAttribute("Hold", "0")
        step.setAttribute("FadeOut", "0")
        step.appendChild(dom.createTextNode(str(chaserid)))
        chaser.appendChild(step)

        # Append the chaser to the Engine element
        engine = dom.getElementsByTagName("Engine")[0]
        engine.appendChild(chaser)

        self.shows[showname]["Functions"].append(attributes)
        showpath = self.shows[showname]["absolute_path"]
        # Write the XML declaration, DOCTYPE declaration, and XML string to the file
        with open(showpath, 'w') as f:
            f.write(dom.toprettyxml(indent="  "))
        return id

    def add_button(self, showname, caption, function_id, key, x=50, y=500):
        dom = self.shows[showname]["dom"]
        # Get the Frame element
        frame = dom.getElementsByTagName("Frame")[0]

        # Create the Button element
        attributes = {"Caption": caption, "ID": str(len(self.shows[showname]["Buttons"])), "Icon": ""}
        button = dom.createElement("Button")
        for i, value in attributes.items():
            button.setAttribute(i, value)

        # Create the WindowState element
        window_state = dom.createElement("WindowState")
        window_state.setAttribute("Visible", "True")
        window_state.setAttribute("X", str(x))
        window_state.setAttribute("Y", str(y))
        window_state.setAttribute("Width", "50")
        window_state.setAttribute("Height", "50")
        button.appendChild(window_state)

        # Create the Appearance element
        appearance = dom.createElement("Appearance")
        for attr in ["FrameStyle", "ForegroundColor", "BackgroundColor", "BackgroundImage", "Font"]:
            child = dom.createElement(attr)
            child.appendChild(dom.createTextNode("Default"))
            appearance.appendChild(child)
        button.appendChild(appearance)

        actiontype = "Toggle"

        if function_id == "blackout":
            actiontype = "Blackout"
            function_id = "4294967295"

        # Create the Function element
        function = dom.createElement("Function")
        function.setAttribute("ID", str(function_id))
        button.appendChild(function)

        # Create the Action element
        action = dom.createElement("Action")
        action.appendChild(dom.createTextNode(actiontype))
        button.appendChild(action)

        # Create the Key element
        key_element = dom.createElement("Key")
        key_element.appendChild(dom.createTextNode(str(key)))
        button.appendChild(key_element)

        # Create the Intensity element
        intensity = dom.createElement("Intensity")
        intensity.setAttribute("Adjust", "False")
        intensity.appendChild(dom.createTextNode("100"))
        button.appendChild(intensity)

        # Append the button to the Frame element
        frame.appendChild(button)

        self.shows[showname]["Buttons"].append(attributes)
        showpath = self.shows[showname]["absolute_path"]
        # Write the XML declaration, DOCTYPE declaration, and XML string to the file
        with open(showpath, 'w') as f:
            f.write(dom.toprettyxml(indent="  "))

    def powershell_script(self, filename):
        # Create the script line
        program_data_path = get_config("program_data_path")
        program_data_path = to_windows_path(program_data_path)
        script_line = f'systemcommand:"{program_data_path}\\AIQLCshows\\play_song.bat" arg:{filename}'
        return script_line
