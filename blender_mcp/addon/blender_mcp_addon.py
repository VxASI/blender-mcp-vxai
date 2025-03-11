# Blender MCP Addon - Socket server for MCP integration
bl_info = {
    "name": "Blender MCP Addon",
    "author": "Your Name",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > MCP Tab",
    "description": "Socket server for Model Context Protocol integration with MCP server",
    "category": "System",
}

import bpy
import socket
import json
import threading
import logging
import base64
import os
import tempfile
from bpy.types import Operator, Panel
from bpy.props import BoolProperty, IntProperty, StringProperty

# Configure logging
log_file = os.path.join(tempfile.gettempdir(), "blender_mcp_addon.log")
logger = logging.getLogger("BlenderMCPAddon")
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler (with error handling)
try:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")
except Exception as e:
    logger.warning(f"Failed to set up file logging: {str(e)}. Falling back to console logging.")

# Server configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9876
BUFFER_SIZE = 8192

# Module-level server instance
_server = None

class MCP_Server:
    """Socket server running inside Blender to handle MCP commands"""
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.thread = None

    def start(self):
        """Start the socket server in a separate thread"""
        global _server
        if self.running:
            logger.info("Server already running")
            return
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.running = True
            
            logger.info(f"Starting MCP server at {self.host}:{self.port}")
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            self.running = False

    def stop(self):
        """Stop the socket server"""
        if not self.running:
            return
        
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                logger.error(f"Error closing server socket: {str(e)}")
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("MCP server stopped")

    def _run(self):
        """Main server loop to accept and handle client connections"""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                logger.info(f"Connection from {addr}")
                threading.Thread(target=self._handle_client, args=(client_socket,), daemon=True).start()
            except Exception as e:
                if self.running:  # Only log if not shutting down intentionally
                    logger.error(f"Server error: {str(e)}")

    def _handle_client(self, client_socket):
        """Handle individual client connection"""
        try:
            while self.running:
                data = client_socket.recv(BUFFER_SIZE)
                if not data:
                    break
                
                try:
                    command = json.loads(data.decode('utf-8'))
                    response = self._process_command(command)
                    client_socket.sendall(json.dumps(response).encode('utf-8'))
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {str(e)}")
                    client_socket.sendall(json.dumps({"status": "error", "message": "Invalid JSON"}).encode('utf-8'))
                except Exception as e:
                    logger.error(f"Error processing command: {str(e)}")
                    client_socket.sendall(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        except Exception as e:
            logger.error(f"Client handling error: {str(e)}")
        finally:
            client_socket.close()

    def _process_command(self, command: dict) -> dict:
        """Process incoming MCP commands and return response"""
        cmd_type = command.get("type")
        params = command.get("params", {})
        
        try:
            if cmd_type == "ping":
                return {"status": "success", "result": {"pong": True}}
            
            elif cmd_type == "get_version_info":
                return {"status": "success", "result": {"version": bpy.app.version_string}}
            
            elif cmd_type == "get_scene_info":
                scene = bpy.context.scene
                return {
                    "status": "success",
                    "result": {
                        "name": scene.name,
                        "frame_start": scene.frame_start,
                        "frame_end": scene.frame_end,
                        "objects": len(scene.objects),
                    }
                }
            
            elif cmd_type == "get_object_list":
                return {
                    "status": "success",
                    "result": [{"name": obj.name, "type": obj.type} for obj in bpy.data.objects]
                }
            
            elif cmd_type == "get_material_list":
                return {
                    "status": "success",
                    "result": [{"name": mat.name} for mat in bpy.data.materials]
                }
            
            elif cmd_type == "render_preview":
                scene = bpy.context.scene
                original_filepath = scene.render.filepath
                scene.render.image_settings.file_format = 'PNG'
                scene.render.filepath = os.path.join(tempfile.gettempdir(), "mcp_preview.png")
                
                bpy.ops.render.render(write_still=True)
                with open(scene.render.filepath, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                
                scene.render.filepath = original_filepath
                try:
                    os.remove(scene.render.filepath)
                except Exception as e:
                    logger.warning(f"Failed to delete preview file: {str(e)}")
                
                return {"status": "success", "result": {"image_data": image_data}}
            
            elif cmd_type == "get_object_info":
                obj_name = params.get("name")
                obj = bpy.data.objects.get(obj_name)
                if not obj:
                    return {"status": "error", "message": f"Object '{obj_name}' not found"}
                return {
                    "status": "success",
                    "result": {
                        "name": obj.name,
                        "type": obj.type,
                        "location": list(obj.location),
                        "rotation": list(obj.rotation_euler),
                        "scale": list(obj.scale),
                    }
                }
            
            elif cmd_type == "create_object":
                type_map = {
                    "CUBE": "primitive_cube_add",
                    "SPHERE": "primitive_uv_sphere_add",
                    "CYLINDER": "primitive_cylinder_add",
                    "PLANE": "primitive_plane_add",
                    "CONE": "primitive_cone_add",
                    "TORUS": "primitive_torus_add",
                    "EMPTY": "object.empty_add",
                    "CAMERA": "object.camera_add",
                    "LIGHT": "object.light_add",
                }
                obj_type = params.get("type", "CUBE").upper()
                op_name = type_map.get(obj_type)
                if not op_name:
                    return {"status": "error", "message": f"Unsupported object type: {obj_type}"}
                
                bpy.ops.object.select_all(action='DESELECT')
                getattr(bpy.ops.mesh if "primitive" in op_name else bpy.ops.object, op_name)()
                obj = bpy.context.active_object
                if params.get("name"):
                    obj.name = params["name"]
                if params.get("location"):
                    obj.location = params["location"]
                if params.get("rotation"):
                    obj.rotation_euler = params["rotation"]
                if params.get("scale"):
                    obj.scale = params["scale"]
                return {"status": "success", "result": {"name": obj.name}}
            
            elif cmd_type == "modify_object":
                obj = bpy.data.objects.get(params.get("name"))
                if not obj:
                    return {"status": "error", "message": f"Object '{params.get('name')}' not found"}
                if "location" in params:
                    obj.location = params["location"]
                if "rotation" in params:
                    obj.rotation_euler = params["rotation"]
                if "scale" in params:
                    obj.scale = params["scale"]
                if "visible" in params:
                    obj.hide_viewport = not params["visible"]
                return {"status": "success", "result": {"modified": obj.name}}
            
            elif cmd_type == "delete_object":
                obj = bpy.data.objects.get(params.get("name"))
                if not obj:
                    return {"status": "error", "message": f"Object '{params.get('name')}' not found"}
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                bpy.ops.object.delete()
                return {"status": "success", "result": {"deleted": params["name"]}}
            
            elif cmd_type == "set_material":
                obj = bpy.data.objects.get(params.get("object_name"))
                if not obj:
                    return {"status": "error", "message": f"Object '{params.get('object_name')}' not found"}
                
                mat_name = params.get("material_name", f"Mat_{obj.name}")
                mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
                if params.get("color"):
                    mat.diffuse_color = (*params["color"], 1.0)  # Add alpha
                if "metallic" in params:
                    mat.metallic = params["metallic"]
                if "roughness" in params:
                    mat.roughness = params["roughness"]
                if "specular" in params:
                    mat.specular_intensity = params["specular"]
                
                if not obj.data.materials:
                    obj.data.materials.append(mat)
                else:
                    obj.data.materials[0] = mat
                return {"status": "success", "result": {"material_name": mat.name}}
            
            elif cmd_type == "render_scene":
                scene = bpy.context.scene
                scene.render.resolution_x = params.get("resolution_x", 800)
                scene.render.resolution_y = params.get("resolution_y", 600)
                if params.get("output_path"):
                    scene.render.filepath = params["output_path"]
                bpy.ops.render.render(write_still=bool(params.get("output_path")))
                return {"status": "success", "result": {"rendered": True, "output_path": scene.render.filepath}}
            
            elif cmd_type == "execute_code":
                exec(params.get("code", ""))
                return {"status": "success", "result": {"executed": True}}
            
            elif cmd_type == "create_text":
                bpy.ops.object.text_add()
                obj = bpy.context.active_object
                text_obj = obj.data
                text_obj.body = params.get("text", "Text")
                if params.get("name"):
                    obj.name = params["name"]
                if params.get("location"):
                    obj.location = params["location"]
                if "extrude" in params:
                    text_obj.extrude = params["extrude"]
                if "bevel_depth" in params:
                    text_obj.bevel_depth = params["bevel_depth"]
                return {"status": "success", "result": {"name": obj.name}}
            
            elif cmd_type == "add_animation":
                obj = bpy.data.objects.get(params.get("object_name"))
                if not obj:
                    return {"status": "error", "message": f"Object '{params.get('object_name')}' not found"}
                prop = params.get("property")
                keyframes = params.get("keyframes", [])
                interp = params.get("interpolation", "BEZIER").upper()
                
                obj.animation_data_create()
                action = bpy.data.actions.new(f"{obj.name}_{prop}_action")
                obj.animation_data.action = action
                fcurve = action.fcurves.new(data_path=prop, index=0)
                
                for kf in keyframes:
                    frame = kf.get("frame")
                    value = kf.get("value")
                    fcurve.keyframe_points.insert(frame, value)
                    kf_point = fcurve.keyframe_points[-1]
                    kf_point.interpolation = interp
                
                return {"status": "success", "result": {"keyframes_added": len(keyframes)}}
            
            elif cmd_type == "import_model":
                file_path = params.get("file_path")
                if file_path.endswith((".obj", ".fbx", ".stl")):
                    if file_path.endswith(".obj"):
                        bpy.ops.import_scene.obj(filepath=file_path)
                    elif file_path.endswith(".fbx"):
                        bpy.ops.import_scene.fbx(filepath=file_path)
                    elif file_path.endswith(".stl"):
                        bpy.ops.import_mesh.stl(filepath=file_path)
                    
                    imported = bpy.context.selected_objects
                    if params.get("location"):
                        for obj in imported:
                            obj.location = params["location"]
                    if params.get("rotation"):
                        for obj in imported:
                            obj.rotation_euler = params["rotation"]
                    if "scale" in params:
                        for obj in imported:
                            obj.scale = (params["scale"],) * 3
                    return {"status": "success", "result": {"imported": True, "object_count": len(imported)}}
                return {"status": "error", "message": f"Unsupported file format: {file_path}"}
            
            else:
                return {"status": "error", "message": f"Unknown command: {cmd_type}"}
        
        except Exception as e:
            logger.error(f"Command '{cmd_type}' failed: {str(e)}")
            return {"status": "error", "message": str(e)}

# Blender operator to start/stop the server
class MCP_OT_Server(Operator):
    bl_idname = "mcp.server_toggle"
    bl_label = "Toggle MCP Server"
    bl_description = "Start or stop the MCP server"
    
    def execute(self, context):
        global _server
        addon_prefs = context.preferences.addons[__name__].preferences
        
        if addon_prefs.server_running:
            if _server:
                _server.stop()
            addon_prefs.server_running = False
            self.report({'INFO'}, "MCP Server stopped")
        else:
            if not _server:
                _server = MCP_Server(addon_prefs.host, addon_prefs.port)
            _server.start()
            addon_prefs.server_running = True
            self.report({'INFO'}, "MCP Server started")
        return {'FINISHED'}

# Addon preferences for settings
class MCP_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    
    server_running: BoolProperty(name="Server Running", default=False)
    port: IntProperty(name="Port", default=DEFAULT_PORT, min=1024, max=65535)
    host: StringProperty(name="Host", default=DEFAULT_HOST)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "host")
        layout.prop(self, "port")
        layout.label(text=f"Server Status: {'Running' if self.server_running else 'Stopped'}")

# Sidebar panel
class MCP_PT_Panel(Panel):
    bl_label = "MCP Server"
    bl_idname = "MCP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MCP"
    
    def draw(self, context):
        layout = self.layout
        addon_prefs = context.preferences.addons[__name__].preferences
        
        layout.operator("mcp.server_toggle", text="Stop Server" if addon_prefs.server_running else "Start Server")
        layout.label(text=f"Host: {addon_prefs.host}")
        layout.label(text=f"Port: {addon_prefs.port}")

# Registration
classes = (MCP_OT_Server, MCP_AddonPreferences, MCP_PT_Panel)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    logger.info("Blender MCP Addon registered")
    global _server
    if not _server:
        addon_prefs = bpy.context.preferences.addons.get(__name__).preferences
        _server = MCP_Server(addon_prefs.host, addon_prefs.port)

def unregister():
    global _server
    if _server and _server.running:
        _server.stop()
    _server = None
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    logger.info("Blender MCP Addon unregistered")

if __name__ == "__main__":
    register()