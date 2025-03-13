import bpy
import json
import logging
import socket
import mathutils
import bmesh
import os
import base64
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

# Configure logging
LOG_DIR = "/tmp"  # Adjust this path as needed for your system
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
        """Process a command, supporting batch operations."""
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
        """Execute a single command using the appropriate handler."""
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
            "subdivide_mesh": self.subdivide_mesh,
            "add_modifier": self.add_modifier,
            "apply_modifier": self.apply_modifier,
            "boolean_operation": self.boolean_operation,
            "select_faces_by_normal": self.select_faces_by_normal,
            "extrude_selected_faces": self.extrude_selected_faces,
            "create_camera": self.create_camera,
            "set_active_camera": self.set_active_camera,
            "render_scene": self.render_scene,
            "get_render_preview": self.get_render_preview,  # New handler
        }
        handler = handlers.get(cmd_type)
        if handler:
            try:
                result = handler(**params)
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error in handler: {str(e)}", exc_info=True)
                return {"status": "error", "message": str(e), "suggestion": "Check parameters or object existence"}
        return {"status": "error", "message": f"Unknown command: {cmd_type}"}

    def get_scene_info(self):
        """Return detailed information about the current Blender scene."""
        scene = bpy.context.scene
        objects = []
        for obj in scene.objects:
            vertex_count = len(obj.data.vertices) if obj.type == 'MESH' else None
            materials = [mat.name for mat in obj.data.materials] if hasattr(obj.data, 'materials') else []
            modifiers = [mod.name for mod in obj.modifiers]
            objects.append({
                "name": obj.name,
                "type": obj.type,
                "location": list(obj.location),
                "rotation": list(obj.rotation_euler),
                "scale": list(obj.scale),
                "vertex_count": vertex_count,
                "materials": materials,
                "modifiers": modifiers
            })
        lights = [{"type": light.type, "location": list(light.location), "strength": light.energy, "color": list(light.color)} for light in scene.objects if light.type == 'LIGHT']
        cameras = [{"name": cam.name, "location": list(cam.location), "rotation": list(cam.rotation_euler), "focal_length": cam.data.lens, "active": cam == scene.camera} for cam in scene.objects if cam.type == 'CAMERA']
        return {
            "name": scene.name,
            "object_count": len(scene.objects),
            "objects": objects,
            "lights": lights,
            "cameras": cameras,
            "frame_range": [scene.frame_start, scene.frame_end],
            "render_settings": {"resolution": [scene.render.resolution_x, scene.render.resolution_y]}
        }

    def get_object_info(self, name):
        """Get information about a specific object."""
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object '{name}' not found")
        return {
            "name": obj.name,
            "type": obj.type,
            "location": list(obj.location),
            "rotation": list(obj.rotation_euler),
            "scale": list(obj.scale),
            "visible": obj.visible_get()
        }

    def create_object(self, type="CUBE", name=None, location=[0, 0, 0], rotation=[0, 0, 0], scale=[1, 1, 1], relative_to=None):
        """Create a new object, optionally positioned relative to another object."""
        if relative_to:
            rel_obj = bpy.data.objects.get(relative_to)
            if not rel_obj:
                raise ValueError(f"Object '{relative_to}' not found")
            location = rel_obj.location + mathutils.Vector(location)
        bpy.ops.object.select_all(action='DESELECT')
        if type == "CUBE":
            bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation, scale=scale)
        elif type == "CYLINDER":
            bpy.ops.mesh.primitive_cylinder_add(location=location, rotation=rotation, scale=scale)
        else:
            raise ValueError(f"Unsupported object type: {type}")
        obj = bpy.context.active_object
        if name:
            obj.name = name
        return {
            "name": obj.name,
            "type": obj.type,
            "location": list(obj.location),
            "rotation": list(obj.rotation_euler),
            "scale": list(obj.scale)
        }

    def modify_object(self, name, location=None, rotation=None, scale=None, visible=None):
        """Modify properties of an existing object."""
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object '{name}' not found")
        if location is not None:
            obj.location = location
        if rotation is not None:
            obj.rotation_euler = rotation
        if scale is not None:
            obj.scale = scale
        if visible is not None:
            obj.hide_viewport = not visible
        return {
            "name": obj.name,
            "location": list(obj.location),
            "rotation": list(obj.rotation_euler),
            "scale": list(obj.scale),
            "visible": obj.visible_get()
        }

    def delete_object(self, name):
        """Delete an object from the scene."""
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object '{name}' not found")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.delete()
        return {"deleted": name}

    def set_material(self, object_name, material_name=None, color=None, material_type="DIFFUSE", **kwargs):
        """Apply or create a material with advanced properties."""
        obj = bpy.data.objects.get(object_name)
        if not obj or not hasattr(obj.data, 'materials'):
            raise ValueError(f"Invalid object: {object_name}")
        mat_name = material_name if material_name else f"Mat_{object_name}"
        mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        tree = mat.node_tree
        nodes = tree.nodes
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        if material_type == "DIFFUSE":
            shader = nodes.new('ShaderNodeBsdfDiffuse')
            if color:
                shader.inputs['Color'].default_value = (*color[:3], 1.0)
        elif material_type == "METALLIC":
            shader = nodes.new('ShaderNodeBsdfPrincipled')
            if color:
                shader.inputs['Base Color'].default_value = (*color[:3], 1.0)
            shader.inputs['Metallic'].default_value = kwargs.get('metallic', 1.0)
            shader.inputs['Roughness'].default_value = kwargs.get('roughness', 0.1)
        elif material_type == "GLASS":
            shader = nodes.new('ShaderNodeBsdfGlass')
            if color:
                shader.inputs['Color'].default_value = (*color[:3], 1.0)
            shader.inputs['IOR'].default_value = kwargs.get('ior', 1.5)
        else:
            raise ValueError(f"Unsupported material type: {material_type}")
        tree.links.new(shader.outputs[0], output.inputs[0])
        if not obj.data.materials:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat
        return {"material_name": mat.name}

    def execute_code(self, code):
        """Execute arbitrary Python code in Blender."""
        try:
            exec(code)
            return {"executed": True}
        except Exception as e:
            raise ValueError(f"Code execution failed: {str(e)}")

    def create_keyframe(self, object_name, frame, properties):
        """Create a keyframe for an object."""
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object '{object_name}' not found")
        bpy.context.scene.frame_set(frame)
        for prop, value in properties.items():
            setattr(obj, prop, value)
            obj.keyframe_insert(data_path=prop, frame=frame)
        return {"keyframed": object_name, "frame": frame}

    def create_terrain(self, name, size, height):
        """Create a simple terrain mesh."""
        bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.name = name
        mod = obj.modifiers.new(name="Displace", type='DISPLACE')
        tex = bpy.data.textures.new("TerrainTex", type='CLOUDS')
        mod.texture = tex
        mod.strength = height
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)
        return {"created": name}

    def export_asset(self, filepath, format="OBJ"):
        """Export the scene or selected objects."""
        if format == "OBJ":
            bpy.ops.export_scene.obj(filepath=filepath, use_selection=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        return {"exported_to": filepath}

    def subdivide_mesh(self, name, cuts):
        """Subdivide a mesh object."""
        obj = bpy.data.objects.get(name)
        if not obj or obj.type != 'MESH':
            raise ValueError(f"Object '{name}' is not a mesh")
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=cuts, use_grid_fill=True)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        return {"subdivided": name, "cuts": cuts}

    def add_modifier(self, name, modifier_type, params):
        """Add a modifier to an object."""
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object '{name}' not found")
        mod = obj.modifiers.new(name=modifier_type, type=modifier_type)
        for key, value in params.items():
            setattr(mod, key, value)
        return {"added_modifier": modifier_type, "to": name}

    def apply_modifier(self, name, modifier_name):
        """Apply a modifier to an object."""
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object '{name}' not found")
        mod = obj.modifiers.get(modifier_name)
        if not mod:
            raise ValueError(f"Modifier '{modifier_name}' not found on object '{name}'")
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)
        return {"applied_modifier": modifier_name, "to": name}

    def boolean_operation(self, obj1, obj2, operation):
        """Perform a boolean operation between two objects."""
        obj = bpy.data.objects.get(obj1)
        cutter = bpy.data.objects.get(obj2)
        if not obj or not cutter:
            raise ValueError(f"Object not found: {obj1} or {obj2}")
        mod = obj.modifiers.new(name="Boolean", type='BOOLEAN')
        mod.operation = operation.upper()
        mod.object = cutter
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)
        return {"boolean_operation": operation, "on": obj1, "with": obj2}

    def select_faces_by_normal(self, name, normal, tolerance):
        """Select faces based on their normal direction."""
        obj = bpy.data.objects.get(name)
        if not obj or obj.type != 'MESH':
            raise ValueError(f"Object '{name}' is not a mesh")
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        normal = mathutils.Vector(normal).normalized()
        selected_faces = [face for face in bm.faces if face.normal.dot(normal) > 1 - tolerance]
        for face in bm.faces:
            face.select = face in selected_faces
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        return {"selected_faces": len(selected_faces)}

    def extrude_selected_faces(self, name, distance):
        """Extrude the selected faces of an object."""
        obj = bpy.data.objects.get(name)
        if not obj or obj.type != 'MESH':
            raise ValueError(f"Object '{name}' not found")
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        selected_faces = [face for face in bm.faces if face.select]
        if not selected_faces:
            raise ValueError("No faces selected for extrusion")
        ret = bmesh.ops.extrude_face_region(bm, geom=selected_faces)
        translate_verts = [v for v in ret['geom'] if isinstance(v, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, vec=(0, 0, distance), verts=translate_verts)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        return {"extruded": name, "distance": distance}

    def create_camera(self, name, location, rotation, focal_length):
        """Create a new camera in the scene."""
        cam_data = bpy.data.cameras.new(name)
        cam = bpy.data.objects.new(name, cam_data)
        cam.location = location
        cam.rotation_euler = rotation
        cam_data.lens = focal_length
        bpy.context.collection.objects.link(cam)
        return {"created_camera": name}

    def set_active_camera(self, name):
        """Set a camera as the active camera."""
        cam = bpy.data.objects.get(name)
        if not cam or cam.type != 'CAMERA':
            raise ValueError(f"Object '{name}' is not a camera")
        bpy.context.scene.camera = cam
        return {"set_active_camera": name}

    def render_scene(self, filepath, resolution):
        """Render the scene to an image file."""
        scene = bpy.context.scene
        if not scene.camera:
            raise ValueError("No active camera set")
        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]
        scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        return {"rendered_to": filepath}

    def get_render_preview(self, resolution=[200, 200]):
        """Generate a low-resolution preview render and return it as base64."""
        scene = bpy.context.scene
        if not scene.camera:
            raise ValueError("No active camera set")
        # Temporarily adjust render settings
        original_res_x = scene.render.resolution_x
        original_res_y = scene.render.resolution_y
        original_filepath = scene.render.filepath
        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]
        temp_filepath = os.path.join(LOG_DIR, "preview.png")
        scene.render.filepath = temp_filepath
        # Render to temporary file
        bpy.ops.render.render(write_still=True)
        # Read and encode the image
        with open(temp_filepath, "rb") as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        # Restore original settings
        scene.render.resolution_x = original_res_x
        scene.render.resolution_y = original_res_y
        scene.render.filepath = original_filepath
        # Clean up temporary file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return {"image_base64": img_data}

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