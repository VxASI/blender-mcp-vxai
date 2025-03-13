import bpy
import json
import logging
import socket
from bpy.props import StringProperty, IntProperty, BoolProperty

bl_info = {
    "name": "Blender MCP",
    "author": "BlenderMCP",
    "version": (0, 1),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > BlenderMCP",
    "description": "Connect Blender to external tools via MCP",
    "category": "Interface",
}

# Configure logging with file output
LOG_DIR = "/tmp"  # Adjust as needed for your system
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

class BlenderMCPServer:
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None
        self.client_socket = None
        self.buffer = b''

    def start(self):
        if self.running:
            logger.info("Server already running")
            return
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            logger.info(f"Attempting to bind to {self.host}:{self.port}")
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.server_socket.setblocking(False)
            self.running = True
            bpy.app.timers.register(self._process_server, persistent=True)
            logger.info(f"MCP server started on {self.host}:{self.port}")
        except socket.error as e:
            logger.error(f"Failed to start server: {str(e)}", exc_info=True)
            self.running = False
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
            raise Exception(f"Failed to bind to {self.host}:{self.port}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error starting server: {str(e)}", exc_info=True)
            self.running = False
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
            raise Exception(f"Unexpected error: {str(e)}")

    def stop(self):
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
                            pass
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
        return 0.1

    def _process_command(self, command):
        cmd_type = command.get("type")
        params = command.get("params", {})
        logger.info(f"Processing command: {cmd_type}, params: {params}")

        if cmd_type == "batch":
            results = []
            for sub_command in params.get("commands", []):
                sub_result = self._process_single_command(sub_command)
                results.append(sub_result)
            return {"status": "success", "result": results}
        return self._process_single_command(command)

    def _process_single_command(self, command):
        cmd_type = command.get("type")
        params = command.get("params", {})
        handlers = {
            "ping": lambda **kwargs: {"pong": True},
            "get_scene_info": self.get_scene_info,
            "get_object_info": self.get_object_info,
            "create_object": self.create_object,
            "modify_object": self.modify_object,
            "delete_object": self.delete_object,
            "set_material": self.set_material,
            "execute_code": self.execute_code,
            "create_keyframe": self.create_keyframe,
            "create_terrain": self.create_terrain,
            "export_asset": self.export_asset,
        }
        handler = handlers.get(cmd_type)
        if handler:
            try:
                result = handler(**params)
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error in handler: {str(e)}", exc_info=True)
                return {"status": "error", "message": str(e)}
        return {"status": "error", "message": f"Unknown command: {cmd_type}"}

    def get_scene_info(self):
        scene = bpy.context.scene
        return {
            "name": scene.name,
            "object_count": len(scene.objects),
            "objects": [{"name": obj.name, "type": obj.type, "location": [obj.location.x, obj.location.y, obj.location.z]} for obj in scene.objects[:10]]
        }

    def get_object_info(self, name):
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object '{name}' not found")
        return {
            "name": obj.name,
            "type": obj.type,
            "location": [obj.location.x, obj.location.y, obj.location.z],
            "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
            "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            "visible": obj.visible_get()
        }

    def create_object(self, type="CUBE", name=None, location=[0, 0, 0], rotation=[0, 0, 0], scale=[1, 1, 1]):
        bpy.ops.object.select_all(action='DESELECT')
        if type == "CUBE":
            bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation, scale=scale)
        elif type == "SPHERE":
            bpy.ops.mesh.primitive_uv_sphere_add(location=location, rotation=rotation, scale=scale)
        elif type == "CYLINDER":
            bpy.ops.mesh.primitive_cylinder_add(location=location, rotation=rotation, scale=scale)
        elif type == "PLANE":
            bpy.ops.mesh.primitive_plane_add(location=location, rotation=rotation, scale=scale)
        elif type == "CONE":
            bpy.ops.mesh.primitive_cone_add(location=location, rotation=rotation, scale=scale)
        elif type == "TORUS":
            bpy.ops.mesh.primitive_torus_add(location=location, rotation=rotation, scale=scale)
        elif type == "EMPTY":
            bpy.ops.object.empty_add(location=location, rotation=rotation, scale=scale)
        elif type == "CAMERA":
            bpy.ops.object.camera_add(location=location, rotation=rotation)
        elif type == "LIGHT":
            bpy.ops.object.light_add(type='POINT', location=location, rotation=rotation, scale=scale)
        else:
            raise ValueError(f"Unsupported type: {type}")
        obj = bpy.context.active_object
        if name:
            obj.name = name
        return {
            "name": obj.name,
            "type": obj.type,
            "location": [obj.location.x, obj.location.y, obj.location.z],
            "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
            "scale": [obj.scale.x, obj.scale.y, obj.scale.z]
        }

    def modify_object(self, name, location=None, rotation=None, scale=None, visible=None):
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object '{name}' not found")
        if location:
            obj.location = location
        if rotation:
            obj.rotation_euler = rotation
        if scale:
            obj.scale = scale
        if visible is not None:
            obj.hide_viewport = not visible
        return {
            "name": obj.name,
            "location": [obj.location.x, obj.location.y, obj.location.z],
            "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
            "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            "visible": obj.visible_get()
        }

    def delete_object(self, name):
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object '{name}' not found")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.delete()
        return {"deleted": name}

    def set_material(self, object_name, material_name=None, color=None):
        obj = bpy.data.objects.get(object_name)
        if not obj or not hasattr(obj.data, 'materials'):
            raise ValueError(f"Invalid object: {object_name}")
        if material_name:
            mat = bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name)
        else:
            mat_name = f"Mat_{object_name}"
            mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
        if not mat.use_nodes:
            mat.use_nodes = True
        principled = mat.node_tree.nodes.get('Principled BSDF') or mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        output = mat.node_tree.nodes.get('Material Output') or mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
        if not principled.outputs[0].links:
            mat.node_tree.links.new(principled.outputs[0], output.inputs[0])
        if color and len(color) >= 3:
            principled.inputs['Base Color'].default_value = (*color[:3], 1.0 if len(color) < 4 else color[3])
        if not obj.data.materials:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat
        return {"material_name": mat.name}

    def create_keyframe(self, object_name, frame, location=None, rotation=None, scale=None):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object '{object_name}' not found")
        bpy.context.scene.frame_set(frame)
        if location:
            obj.location = location
            obj.keyframe_insert(data_path="location", frame=frame)
        if rotation:
            obj.rotation_euler = rotation
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        if scale:
            obj.scale = scale
            obj.keyframe_insert(data_path="scale", frame=frame)
        return {"frame": frame, "object_name": object_name}

    def create_terrain(self, name="Terrain", size=None, height=1.0, subdivisions=64):
        size = size or [10.0, 10.0]
        bpy.ops.mesh.primitive_plane_add(size=size[0], location=[0, 0, 0])
        obj = bpy.context.active_object
        obj.name = name
        # Subdivide for terrain detail
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=subdivisions)
        bpy.ops.object.mode_set(mode='OBJECT')
        # Add displace modifier
        bpy.ops.object.modifier_add(type='DISPLACE')
        displace = obj.modifiers["Displace"]
        displace.strength = height
        # Add a noise texture for displacement
        tex = bpy.data.textures.new(name=f"{name}_Noise", type='CLOUDS')
        displace.texture = tex
        displace.texture_coords = 'LOCAL'
        bpy.ops.object.modifier_apply(modifier="Displace")
        return {"name": obj.name, "size": size, "height": height}

    def export_asset(self, object_name, export_path, format="GLTF"):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object '{object_name}' not found")
        # Deselect all and select the target object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        if format == "GLTF":
            bpy.ops.export_scene.gltf(filepath=export_path, export_format='GLTF_SEPARATE', use_selection=True)
        elif format == "FBX":
            bpy.ops.export_scene.fbx(filepath=export_path, use_selection=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        return {"path": export_path, "format": format}

    def execute_code(self, code):
        namespace = {"bpy": bpy}
        exec(code, namespace)
        return {"executed": True}

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
        name="Port", default=9876, min=1024, max=65535)
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