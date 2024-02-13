from xml.dom.minidom import parse, Document
import shutil

class QXWHandler:
    def __init__(self, file):
        self.file = file
        self.dom = parse(file)
        self.shows = {}

    @staticmethod
    def remove_whitespace_nodes(node, unlink=False):
        """Removes all of the whitespace-only text descendants of node.
        If unlink is True, also unlinks the nodes from their parents."""
        remove_list = []
        for child in node.childNodes:
            if child.nodeType == node.TEXT_NODE and not child.data.strip():
                remove_list.append(child)
            elif child.hasChildNodes():
                QXWHandler.remove_whitespace_nodes(child, unlink)
        for node in remove_list:
            node.parentNode.removeChild(node)
            if unlink:
                node.unlink()

    def create_copy(self, showname):
        shutil.copy2(self.file, f"{showname}.qxw")
        self.shows[showname] = {"Functions": []}        
    def add_element(self, showname, parent_element_name, element_name, attributes={}, text=None, child_elements=[]):
        # Create the new element
        element = self.dom.createElement(element_name)
        for key, value in attributes.items():
            element.setAttribute(key, value)
        if text is not None:
            element.appendChild(self.dom.createTextNode(text))

        # Add child elements
        for child_element in child_elements:
            element.appendChild(child_element)

        # Find the parent element and append the new element to it
        parent_element = self.dom.getElementsByTagName(parent_element_name)[0]
        parent_element.appendChild(element)

        # Remove whitespace nodes
        QXWHandler.remove_whitespace_nodes(self.dom)

        # Write the XML declaration, DOCTYPE declaration, and XML string to the file
        with open(f"{showname}.qxw", 'w') as f:
            f.write(self.dom.toprettyxml(indent="  "))

    def add_script(self, showname, script, function_name):
        # Create the Speed, Direction, and RunOrder elements
        speed = self.dom.createElement('Speed')
        speed.setAttribute('FadeIn', '0')
        speed.setAttribute('FadeOut', '0')
        speed.setAttribute('Duration', '0')

        direction = self.dom.createElement('Direction')
        direction.appendChild(self.dom.createTextNode('Forward'))

        run_order = self.dom.createElement('RunOrder')
        run_order.appendChild(self.dom.createTextNode('Loop'))

        # Create the Command elements
        commands = []
        for command in script:
            command_elem = self.dom.createElement('Command')
            command_elem.appendChild(self.dom.createTextNode(command))
            commands.append(command_elem)

        function_info = {'ID': str(len(self.shows[showname]["Functions"])), 'Type': 'Script', 'Name': str(function_name)}

        # Add the Function element to the XML file
        self.add_element(showname, 'Engine', 'Function', function_info, None, [speed, direction, run_order] + commands)
        self.shows[showname]["Functions"].append(function_info)

# Create a QXWHandler for the test file
qxw_handler = QXWHandler('TEST.qxw')
new_qxw_handler = qxw_handler.create_copy('NEW')

# Define a script and function info
script = ['wait%3A0%20%2F%2FWait%20for%20pause0', 'blackout%3Aon%20%2F%2FBlackout%20for%204.534883720930233%20seconds']
function_name = "New script 0"

# Add the script to the file
qxw_handler.add_script("NEW", script, function_name)

script = ['wait%3A0%20%2F%2FWait%20for%20pause0', 'blackout%3Aon%20%2F%2FBlackout%20for%204.534883720930233%20seconds']
function_name = "New script 1"
qxw_handler.add_script("NEW", script, function_name)