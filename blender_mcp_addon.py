import bpy
import json
import logging
import socket
import os
import time
import random
from bpy.props import IntProperty, BoolProperty
import base64
import math

bl_info = {
    "name": "Blender MCP",
    "author": "BlenderMCP",
    "version": (0, 3),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > BlenderMCP",
    "description": "MCP integration for dynamic Blender scene manipulation",
    "category": "Interface",
}

# Configure logging
LOG_DIR = "/tmp"  # Adjust this path as needed
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "blender_mcp_addon.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BlenderMCPAddon")

# Global history to track actions
_action_history = []

class BlenderMCPServer:
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None
        self.client_socket = None
        self.buffer = b''

    def start(self):
        """Start the MCP server to listen for connections."""
        if self.running:
            logger.info("Server already running")
            return
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.server_socket.setblocking(False)
            self.running = True
            bpy.app.timers.register(self._process_server, persistent=True)
            logger.info(f"MCP server started on {self.host}:{self.port}")
        except socket.error as e:
            logger.error(f"Failed to start server: {str(e)}")
            self.running = False
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
            raise Exception(f"Failed to bind to {self.host}:{self.port}: {str(e)}")

    def stop(self):
        """Stop the MCP server and clean up resources."""
        if not self.running:
            logger.info("Server not running")
            return
        self.running = False
        if bpy.app.timers.is_registered(self._process_server):
            bpy.app.timers.unregister(self._process_server)
        if self.server_socket:
            self.server_socket.close()
        if self.client_socket:
            self.client_socket.close()
        self.server_socket = None
        self.client_socket = None
        self.buffer = b''
        logger.info("MCP server stopped")

    def _process_server(self):
        """Handle incoming connections and commands in a non-blocking manner."""
        if not self.running:
            return None
        try:
            if not self.client_socket and self.server_socket:
                try:
                    self.client_socket, addr = self.server_socket.accept()
                    self.client_socket.setblocking(False)
                    logger.info(f"Connected to client: {addr}")
                except BlockingIOError:
                    pass
            if self.client_socket:
                try:
                    data = self.client_socket.recv(8192)
                    if data:
                        self.buffer += data
                        try:
                            command = json.loads(self.buffer.decode('utf-8'))
                            self.buffer = b''
                            response = self._process_command(command)
                            self.client_socket.sendall(json.dumps(response).encode('utf-8'))
                        except json.JSONDecodeError:
                            pass  # Wait for more data
                    else:
                        logger.info("Client disconnected")
                        self.client_socket.close()
                        self.client_socket = None
                        self.buffer = b''
                except BlockingIOError:
                    pass
                except Exception as e:
                    logger.error(f"Error with client: {str(e)}")
                    if self.client_socket:
                        self.client_socket.close()
                        self.client_socket = None
                    self.buffer = b''
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        return 0.1  # Check every 0.1 seconds

    def _process_command(self, command):
        """Process a command received from the MCP server."""
        cmd_type = command.get("type")
        params = command.get("params", {})
        logger.info(f"Processing command: {cmd_type}, params: {params}")

        handlers = {
            "get_scene_info": self.get_scene_info,
            "run_script": self.run_script
        }
        handler = handlers.get(cmd_type)
        if handler:
            try:
                result = handler(**params)
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error in handler: {str(e)}", exc_info=True)
                return {"status": "error", "message": str(e), "suggestion": "Check parameters or script syntax"}
        return {"status": "error", "message": f"Unknown command: {cmd_type}"}

    def get_scene_info(self, filters=None, properties=None, sub_object_data=None, limit=None, offset=0, timeout=5.0):
        """Return detailed information about the Blender scene with filtering, pagination, and timeout control."""
        start_time = time.time()
        scene = bpy.context.scene
        objects = list(scene.objects)  # Convert to list for slicing

        # Apply filters
        filters = filters or {}
        if "type" in filters:
            objects = [obj for obj in objects if obj.type == filters["type"]]
        if "name_contains" in filters:
            objects = [obj for obj in objects if filters["name_contains"] in obj.name]
        if "spatial_bounds" in filters:
            bounds = filters["spatial_bounds"]
            min_bounds = bounds.get("min", [-float('inf')] * 3)
            max_bounds = bounds.get("max", [float('inf')] * 3)
            objects = [
                obj for obj in objects
                if all(min_bounds[i] <= obj.location[i] <= max_bounds[i] for i in range(3))
            ]

        # Apply pagination
        total_count = len(objects)
        objects = objects[offset:offset + (limit or len(objects))]

        # Default properties
        properties = properties or ["name", "type", "location"]
        sub_object_data = sub_object_data or {}

        # Prepare scene data
        scene_data = {"objects": [], "cameras": [], "lights": [], "history": _action_history[-10:]}

        for obj in objects:
            # Check timeout
            if time.time() - start_time > timeout:
                return {
                    "status": "timeout",
                    "partial_data": scene_data,
                    "message": "Operation timed out. Try applying more specific filters for better performance.",
                    "total_count": total_count,
                    "processed_count": len(scene_data["objects"]) + len(scene_data["cameras"]) + len(scene_data["lights"])
                }

            obj_data = {}
            for prop in properties:
                if time.time() - start_time > timeout:
                    break  # Exit early if timeout is exceeded
                if prop == "name":
                    obj_data["name"] = obj.name
                elif prop == "type":
                    obj_data["type"] = obj.type
                elif prop == "location":
                    obj_data["location"] = list(obj.location)
                elif prop == "rotation":
                    obj_data["rotation"] = list(obj.rotation_euler)
                elif prop == "scale":
                    obj_data["scale"] = list(obj.scale)
                elif prop == "vertex_count" and obj.type == "MESH":
                    obj_data["vertex_count"] = len(obj.data.vertices)
                elif prop == "face_count" and obj.type == "MESH":
                    obj_data["face_count"] = len(obj.data.polygons)
                elif prop == "vertices" and obj.type == "MESH":
                    vertex_opts = sub_object_data.get("vertices", {})
                    sample_rate = vertex_opts.get("sample_rate", 1.0)
                    max_count = vertex_opts.get("max_count", float('inf'))
                    vertices = [list(v.co) for v in obj.data.vertices]
                    if sample_rate < 1.0:
                        vertices = [v for i, v in enumerate(vertices) if random.random() < sample_rate]
                    if len(vertices) > max_count:
                        vertices = vertices[:int(max_count)]
                    obj_data["vertices"] = vertices
                elif prop == "modifiers":
                    obj_data["modifiers"] = [mod.name for mod in obj.modifiers]

            # Categorize objects
            if obj.type == "CAMERA":
                scene_data["cameras"].append(obj_data)
            elif obj.type == "LIGHT":
                scene_data["lights"].append(obj_data)
            else:
                scene_data["objects"].append(obj_data)

        return scene_data

    def run_script(self, script: str):
        """Execute a Python script in Blender."""
        global _action_history
        try:
            # Decode base64-encoded script
            script_decoded = base64.b64decode(script).decode('utf-8')
            # Define globals with pre-imported modules
            script_globals = {'bpy': bpy, 'math': math, 'random': random}
            # Execute script
            exec(script_decoded, script_globals, locals())
            _action_history.append(f"Executed script: {script_decoded[:50]}...")
            return {"message": "Script executed successfully"}
        except Exception as e:
            _action_history.append(f"Script execution failed: {str(e)}")
            raise Exception(f"Script execution failed: {str(e)}")

# UI Panel
class BLENDERMCP_PT_Panel(bpy.types.Panel):
    bl_label = "Blender MCP"
    bl_idname = "BLENDERMCP_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BlenderMCP'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "blendermcp_port")
        if not scene.blendermcp_server_running:
            layout.operator("blendermcp.start_server", text="Start MCP Server")
        else:
            layout.operator("blendermcp.stop_server", text="Stop MCP Server")
            layout.label(text=f"Running on port {scene.blendermcp_port}")

# Operators
class BLENDERMCP_OT_StartServer(bpy.types.Operator):
    bl_idname = "blendermcp.start_server"
    bl_label = "Start MCP Server"
    bl_description = "Start the MCP server"

    def execute(self, context):
        scene = context.scene
        try:
            if not hasattr(bpy.types, "blendermcp_server") or not bpy.types.blendermcp_server:
                bpy.types.blendermcp_server = BlenderMCPServer(port=scene.blendermcp_port)
            bpy.types.blendermcp_server.start()
            scene.blendermcp_server_running = True
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start MCP server: {str(e)}")
            scene.blendermcp_server_running = False
            return {'CANCELLED'}
        return {'FINISHED'}

class BLENDERMCP_OT_StopServer(bpy.types.Operator):
    bl_idname = "blendermcp.stop_server"
    bl_label = "Stop MCP Server"
    bl_description = "Stop the MCP server"

    def execute(self, context):
        scene = context.scene
        if hasattr(bpy.types, "blendermcp_server") and bpy.types.blendermcp_server:
            bpy.types.blendermcp_server.stop()
            del bpy.types.blendermcp_server
        scene.blendermcp_server_running = False
        return {'FINISHED'}

# Registration
def register():
    bpy.types.Scene.blendermcp_port = IntProperty(
        name="Port", default=9876, min=1024, max=65535, description="Port for MCP server")
    bpy.types.Scene.blendermcp_server_running = BoolProperty(default=False)
    bpy.utils.register_class(BLENDERMCP_PT_Panel)
    bpy.utils.register_class(BLENDERMCP_OT_StartServer)
    bpy.utils.register_class(BLENDERMCP_OT_StopServer)
    logger.info("BlenderMCP addon registered")

def unregister():
    if hasattr(bpy.types, "blendermcp_server") and bpy.types.blendermcp_server:
        bpy.types.blendermcp_server.stop()
        del bpy.types.blendermcp_server
    bpy.utils.unregister_class(BLENDERMCP_PT_Panel)
    bpy.utils.unregister_class(BLENDERMCP_OT_StartServer)
    bpy.utils.unregister_class(BLENDERMCP_OT_StopServer)
    del bpy.types.Scene.blendermcp_port
    del bpy.types.Scene.blendermcp_server_running
    logger.info("BlenderMCP addon unregistered")

if __name__ == "__main__":
    register()