import shutil
import os
from xml.dom.minidom import parse
from src.utils.config import get_config, to_windows_path

class QLCManager:
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
                QLCManager.remove_whitespace_nodes(child, unlink)
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
        print(f"----Created a copy of template to {absolute_path}")

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
        QLCManager.remove_whitespace_nodes(dom)
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
        print(f"----Generating QLC show {showname} with {len(show_scripts)} scripts")
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
        print(f"----Generated QLC show to {showpath}")

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
        script_line = f'systemcommand:"{program_data_path}\\AIQLCshows\\play_song_modified.bat" arg:{filename}'
        return script_line

    def merge_shows(self, target_showname, source_shows, max_workers=4):
        """
        Creates a merged show file that combines scripts from multiple source shows.
        Uses multithreading to process source shows in parallel.
        
        Args:
            target_showname (str): Name of the target show to create
            source_shows (list): List of dictionaries with 'name' and 'duration' keys for each show
            max_workers (int): Maximum number of worker threads to use (default: 4)
        """
        import threading
        import concurrent.futures
        from queue import Queue
        
        # Extract just the show names from the source shows
        source_shownames = [show['name'] for show in source_shows]
        durations = {show['name']: show['duration'] for show in source_shows}
        
        # Copy template file
        self.create_copy(target_showname)
        
        # Get DOM once
        dom = self.shows[target_showname]["dom"]
        engine = dom.getElementsByTagName("Engine")[0]
        showpath = self.shows[target_showname]["absolute_path"]
        
        # Create synchronization primitives
        function_id_lock = threading.Lock()
        dom_lock = threading.Lock()
        
        # Queue for collecting results
        collections_queue = Queue()
        
        # Counter for function IDs
        next_function_id = len(self.shows[target_showname]["Functions"])
        
        def get_next_function_id():
            """Thread-safe function to get the next available function ID"""
            nonlocal next_function_id
            with function_id_lock:
                function_id = str(next_function_id)
                next_function_id += 1
                return function_id
        
        def process_show(source_name):
            """Process a single source show and create its collection"""
            try:
                # Load the source show DOM
                source_path = f"{self.destination_folder}/{source_name}.qxw"
                if not os.path.exists(source_path):
                    print(f"Warning: Source show '{source_name}' not found at {source_path}, skipping.")
                    return None
                
                # Find all eligible scripts in the source show
                source_dom = parse(source_path)
                scripts_to_copy = []
                
                # Get all functions from the source DOM
                functions = source_dom.getElementsByTagName("Function")
                for function in functions:
                    # Check if it's a script
                    if function.getAttribute("Type") == "Script":
                        name = function.getAttribute("Name")
                        
                        # Check if name starts with a number or is "pauses"
                        if (name and (name[0].isdigit() or name == "pauses")):
                            # Get all commands for this script
                            commands = []
                            command_elements = function.getElementsByTagName("Command")
                            for cmd in command_elements:
                                if cmd.firstChild and cmd.firstChild.nodeType == cmd.firstChild.TEXT_NODE:
                                    commands.append(cmd.firstChild.data)
                            
                            scripts_to_copy.append({
                                "name": name, 
                                "commands": commands
                            })
                
                if not scripts_to_copy:
                    print(f"No eligible scripts found in '{source_name}'")
                    return None
                
                print(f"Found {len(scripts_to_copy)} scripts to copy from '{source_name}'")
                
                # Create new scripts in the target show with prefixed names
                new_script_ids = []
                
                # Process scripts in batches to minimize lock contention
                with dom_lock:
                    # First add the powershell script to play this show's audio
                    ps_script_id = get_next_function_id()
                    ps_commands = [self.powershell_script(f"{source_name}.wav")]
                    
                    # Create the powershell script element
                    speed = dom.createElement('Speed')
                    speed.setAttribute('FadeIn', '0')
                    speed.setAttribute('FadeOut', '0')
                    speed.setAttribute('Duration', '0')
                    
                    direction = dom.createElement('Direction')
                    direction.appendChild(dom.createTextNode('Forward'))
                    
                    run_order = dom.createElement('RunOrder')
                    run_order.appendChild(dom.createTextNode('Loop'))
                    
                    # Create the Command elements for powershell
                    command_elems = []
                    for command in ps_commands:
                        command_elem = dom.createElement('Command')
                        command_elem.appendChild(dom.createTextNode(command))
                        command_elems.append(command_elem)
                    
                    # Add powershell script to DOM
                    function_name = f"{source_name}_powershell_script"
                    ps_function = dom.createElement('Function')
                    ps_function.setAttribute('ID', ps_script_id)
                    ps_function.setAttribute('Type', 'Script')
                    ps_function.setAttribute('Name', function_name)
                    
                    ps_function.appendChild(speed.cloneNode(True))
                    ps_function.appendChild(direction.cloneNode(True))
                    ps_function.appendChild(run_order.cloneNode(True))
                    
                    for cmd in command_elems:
                        ps_function.appendChild(cmd)
                    
                    engine.appendChild(ps_function)
                    new_script_ids.append(ps_script_id)
                    self.shows[target_showname]["Functions"].append({
                        'ID': ps_script_id, 
                        'Type': 'Script', 
                        'Name': function_name
                    })
                    
                    # Create scripts for all other commands
                    for script in scripts_to_copy:
                        script_id = get_next_function_id()
                        new_name = f"{source_name}_{script['name']}"
                        
                        # Create script function
                        script_function = dom.createElement('Function')
                        script_function.setAttribute('ID', script_id)
                        script_function.setAttribute('Type', 'Script')
                        script_function.setAttribute('Name', new_name)
                        
                        script_function.appendChild(speed.cloneNode(True))
                        script_function.appendChild(direction.cloneNode(True))
                        script_function.appendChild(run_order.cloneNode(True))
                        
                        # Add commands
                        for command in script["commands"]:
                            cmd_elem = dom.createElement('Command')
                            cmd_elem.appendChild(dom.createTextNode(command))
                            script_function.appendChild(cmd_elem)
                        
                        engine.appendChild(script_function)
                        new_script_ids.append(script_id)
                        self.shows[target_showname]["Functions"].append({
                            'ID': script_id, 
                            'Type': 'Script', 
                            'Name': new_name
                        })
                    
                    # Create a collection for this source show's scripts
                    collection_name = f"{source_name}_collection"
                    collection_id = get_next_function_id()
                    
                    # Create collection function
                    collection = dom.createElement('Function')
                    collection.setAttribute('ID', collection_id)
                    collection.setAttribute('Type', 'Collection')
                    collection.setAttribute('Name', collection_name)
                    
                    # Add steps to collection
                    for i, script_id in enumerate(new_script_ids):
                        step = dom.createElement("Step")
                        step.setAttribute("Number", str(i))
                        step.appendChild(dom.createTextNode(str(script_id)))
                        collection.appendChild(step)
                    
                    engine.appendChild(collection)
                    
                    self.shows[target_showname]["Functions"].append({
                        'ID': collection_id, 
                        'Type': 'Collection', 
                        'Name': collection_name
                    })
                
                collection_info = {
                    "id": collection_id,
                    "name": collection_name,
                    "source_name": source_name,
                    "duration": durations[source_name]
                }
                
                collections_queue.put(collection_info)
                print(f"Created collection '{collection_name}' with {len(new_script_ids)} scripts")
                return collection_info
            
            except Exception as e:
                print(f"Error processing show {source_name}: {str(e)}")
                return None
        
        # Process source shows in parallel
        all_collections = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all processing tasks
            future_to_show = {
                executor.submit(process_show, source_name): source_name 
                for source_name in source_shownames
            }
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(future_to_show):
                source_name = future_to_show[future]
                try:
                    result = future.result()
                    if result:
                        all_collections.append(result)
                except Exception as e:
                    print(f"Error processing show {source_name}: {str(e)}")
        
        # Create chained scripts if we have collections
        if all_collections:
            self._create_chained_scripts_mt(
                target_showname, all_collections, dom, engine, get_next_function_id
            )
            
            # Write the final DOM to file once
            with open(showpath, 'w') as f:
                f.write(dom.toprettyxml(indent="  "))
        
        print(f"Successfully merged scripts from {len(source_shownames)} shows into '{target_showname}'.")

    def _create_chained_scripts_mt(self, target_showname, all_collections, dom, engine, id_generator):
        """Create scripts that chain together in sequence for playback (multithreaded version)"""
        # Randomize the order
        import random
        random.shuffle(all_collections)
        
        # Create script data for each collection
        script_infos = []
        for i, collection in enumerate(all_collections):
            script_infos.append({
                "name": f"Play_{collection['source_name']}",
                "collection": collection,
                "index": i
            })
        
        # Create all scripts at once with complete commands
        script_ids = []
        for i, script_info in enumerate(script_infos):
            collection = script_info["collection"]
            
            # Build initial command list for this script
            commands = [
                f"startfunction:{collection['id']} // Start {collection['name']}",
                f"wait:{collection['duration']} // Wait for {collection['name']} to finish"
            ]
            
            # Create the script to get its ID
            function_id = id_generator()
            script_name = script_info["name"]
            
            # Create script function
            script_function = dom.createElement('Function')
            script_function.setAttribute('ID', function_id)
            script_function.setAttribute('Type', 'Script')
            script_function.setAttribute('Name', script_name)
            
            # Add standard elements
            speed = dom.createElement('Speed')
            speed.setAttribute('FadeIn', '0')
            speed.setAttribute('FadeOut', '0')
            speed.setAttribute('Duration', '0')
            script_function.appendChild(speed)
            
            direction = dom.createElement('Direction')
            direction.appendChild(dom.createTextNode('Forward'))
            script_function.appendChild(direction)
            
            run_order = dom.createElement('RunOrder')
            run_order.appendChild(dom.createTextNode('Loop'))
            script_function.appendChild(run_order)
            
            # Add commands
            for command in commands:
                cmd_elem = dom.createElement('Command')
                cmd_elem.appendChild(dom.createTextNode(command))
                script_function.appendChild(cmd_elem)
            
            # Add to engine
            engine.appendChild(script_function)
            
            # Track in shows dictionary
            self.shows[target_showname]["Functions"].append({
                'ID': function_id, 
                'Type': 'Script', 
                'Name': script_name
            })
            
            script_ids.append(function_id)
        
        # Now that we have all IDs, update each script with chain command
        for i, script_id in enumerate(script_ids):
            next_index = (i + 1) % len(script_ids)
            next_script_id = script_ids[next_index]
            next_script_name = script_infos[next_index]["name"]
            
            # Find the script function
            for function in engine.getElementsByTagName("Function"):
                if function.getAttribute("Type") == "Script" and function.getAttribute("ID") == script_id:
                    # Add chain to next script command
                    next_script_command = dom.createElement("Command")
                    next_script_command.appendChild(dom.createTextNode(
                        f"startfunction:{next_script_id} // Chain to {next_script_name}"
                    ))
                    function.appendChild(next_script_command)
                    
                    # Add safety wait
                    long_wait_command = dom.createElement("Command")
                    long_wait_command.appendChild(dom.createTextNode(
                        "wait:10800000 // Safety wait (3 hours)"
                    ))
                    function.appendChild(long_wait_command)
                    break
        
        # Create buttons for the first few scripts
        self._create_script_buttons_mt(target_showname, script_ids, script_infos, dom)
        
        print(f"Created {len(script_ids)} chained scripts with safety waits")
        print(f"The scripts form a continuous loop - the last one starts the first one again")

    def _create_script_buttons_mt(self, target_showname, script_ids, script_infos, dom):
        """Create buttons for the scripts (multithreaded version)"""
        num_buttons = min(5, len(script_ids))
        
        # Get the Frame element
        frame = dom.getElementsByTagName("Frame")[0]
        
        for i in range(num_buttons):
            # Spread buttons horizontally
            x_pos = 50 + (i * 60)
            
            # Get the script ID
            script_id = script_ids[i]
            
            # Create button info
            if i == 0:
                button_text = "PLAY ALL"
                key = "P"  # P key shortcut
            else:
                button_text = f"#{i+1}"
                key = str(i)  # Number keys
            
            # Create the Button element directly
            button_id = len(self.shows[target_showname]["Buttons"])
            attributes = {"Caption": button_text, "ID": str(button_id), "Icon": ""}
            button = dom.createElement("Button")
            for attr_name, attr_value in attributes.items():
                button.setAttribute(attr_name, attr_value)
            
            # Create the WindowState element
            window_state = dom.createElement("WindowState")
            window_state.setAttribute("Visible", "True")
            window_state.setAttribute("X", str(x_pos))
            window_state.setAttribute("Y", "50")
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
            
            # Create the Function element
            function_elem = dom.createElement("Function")
            function_elem.setAttribute("ID", str(script_id))
            button.appendChild(function_elem)
            
            # Create the Action element
            action = dom.createElement("Action")
            action.appendChild(dom.createTextNode("Toggle"))
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
            
            self.shows[target_showname]["Buttons"].append(attributes)
        
        # Create a blackout button
        blackout_id = len(self.shows[target_showname]["Buttons"])
        attributes = {"Caption": "BLACKOUT", "ID": str(blackout_id), "Icon": ""}
        button = dom.createElement("Button")
        for attr_name, attr_value in attributes.items():
            button.setAttribute(attr_name, attr_value)
        
        # Window state for blackout
        window_state = dom.createElement("WindowState")
        window_state.setAttribute("Visible", "True")
        window_state.setAttribute("X", "50")
        window_state.setAttribute("Y", "110")
        window_state.setAttribute("Width", "50")
        window_state.setAttribute("Height", "50")
        button.appendChild(window_state)
        
        # Rest of blackout button properties
        appearance = dom.createElement("Appearance")
        for attr in ["FrameStyle", "ForegroundColor", "BackgroundColor", "BackgroundImage", "Font"]:
            child = dom.createElement(attr)
            child.appendChild(dom.createTextNode("Default"))
            appearance.appendChild(child)
        button.appendChild(appearance)
        
        function_elem = dom.createElement("Function")
        function_elem.setAttribute("ID", "4294967295")  # Blackout function ID
        button.appendChild(function_elem)
        
        action = dom.createElement("Action")
        action.appendChild(dom.createTextNode("Blackout"))
        button.appendChild(action)
        
        key_element = dom.createElement("Key")
        key_element.appendChild(dom.createTextNode("B"))
        button.appendChild(key_element)
        
        intensity = dom.createElement("Intensity")
        intensity.setAttribute("Adjust", "False")
        intensity.appendChild(dom.createTextNode("100"))
        button.appendChild(intensity)
        
        frame.appendChild(button)
        self.shows[target_showname]["Buttons"].append(attributes)

    def _create_collection(self, showname, collection_name, function_ids):
        """
        Creates a collection containing the specified function IDs.
        
        Args:
            showname (str): Name of the show
            collection_name (str): Name for the collection
            function_ids (list): List of function IDs to include in the collection
        
        Returns:
            str: ID of the created collection
        """
        dom = self.shows[showname]["dom"]
        
        # Create the collection
        collection_id = str(len(self.shows[showname]["Functions"]))
        attributes = {"ID": collection_id, "Type": "Collection", "Name": collection_name}
        
        # Add the basic collection element
        self.add_element(showname, "Engine", "Function", attributes)
        
        # Get the newly created collection
        engine = dom.getElementsByTagName("Engine")[0]
        collection = engine.getElementsByTagName("Function")[-1]
        
        # Add the function IDs as steps
        for i, function_id in enumerate(function_ids):
            step = dom.createElement("Step")
            step.setAttribute("Number", str(i))
            step.appendChild(dom.createTextNode(str(function_id)))
            collection.appendChild(step)
        
        self.shows[showname]["Functions"].append(attributes)
        showpath = self.shows[showname]["absolute_path"]
        
        # Write the updated DOM to file
        with open(showpath, 'w') as f:
            f.write(dom.toprettyxml(indent="  "))
        
        return collection_id