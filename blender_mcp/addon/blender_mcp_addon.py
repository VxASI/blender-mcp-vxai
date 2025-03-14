import bpy
import json
import logging
import socket
import os
from bpy.props import IntProperty, BoolProperty

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

    def get_scene_info(self):
        """Return detailed information about the current Blender scene."""
        scene = bpy.context.scene
        objects = []
        for obj in scene.objects:
            vertex_count = len(obj.data.vertices) if obj.type == 'MESH' else None
            face_count = len(obj.data.polygons) if obj.type == 'MESH' else None
            modifiers = [mod.name for mod in obj.modifiers] if obj.modifiers else []
            objects.append({
                "name": obj.name,
                "type": obj.type,
                "location": list(obj.location),
                "rotation": list(obj.rotation_euler),
                "scale": list(obj.scale),
                "vertex_count": vertex_count,
                "face_count": face_count,
                "modifiers": modifiers
            })
        cameras = [
            {
                "name": cam.name,
                "location": list(cam.location),
                "rotation": list(cam.rotation_euler)
            } for cam in scene.objects if cam.type == 'CAMERA'
        ]
        lights = [
            {
                "name": light.name,
                "type": light.data.type,
                "location": list(light.location),
                "intensity": light.data.energy,
                "color": list(light.data.color)
            } for light in scene.objects if light.type == 'LIGHT'
        ]
        return {
            "objects": objects,
            "cameras": cameras,
            "lights": lights,
            "history": _action_history[-10:]  # Last 10 actions for brevity
        }

    def run_script(self, script: str):
        """Execute a Python script in Blender."""
        global _action_history
        try:
            # Execute the script in Blender's Python environment
            exec(script, globals(), locals())
            _action_history.append(f"Executed script: {script[:50]}...")
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