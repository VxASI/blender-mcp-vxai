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
import bmesh
from mathutils import Vector, Matrix
from typing import Dict, Any, List, Optional  # Added List import

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
            "run_script": self.run_script,
            "edit_mesh": self.edit_mesh
        }
        handler = handlers.get(cmd_type)
        if handler:
            try:
                result = handler(**params)
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error in handler: {str(e)}", exc_info=True)
                return {"status": "error", "message": str(e), "suggestion": "Check parameters or ensure object is a mesh. Try selecting vertices first if applicable."}
        return {"status": "error", "message": f"Unknown command: {cmd_type}"}

    def get_scene_info(self, filters=None, properties=None, sub_object_data=None, limit=None, offset=0, timeout=5.0):
        """Return detailed information about the Blender scene with filtering, pagination, and timeout control."""
        start_time = time.time()
        scene = bpy.context.scene
        objects = list(scene.objects)

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

        total_count = len(objects)
        objects = objects[offset:offset + (limit or len(objects))]

        properties = properties or ["name", "type", "location"]
        sub_object_data = sub_object_data or {}

        scene_data = {"objects": [], "cameras": [], "lights": [], "history": _action_history[-10:]}

        for obj in objects:
            if time.time() - start_time > timeout:
                return {
                    "status": "timeout",
                    "partial_data": scene_data,
                    "message": "Operation timed out.",
                    "total_count": total_count,
                    "processed_count": len(scene_data["objects"]) + len(scene_data["cameras"]) + len(scene_data["lights"])
                }

            obj_data = {}
            for prop in properties:
                if time.time() - start_time > timeout:
                    break
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
            script_decoded = base64.b64decode(script).decode('utf-8')
            script_globals = {'bpy': bpy, 'math': math, 'random': random}
            exec(script_decoded, script_globals, locals())
            _action_history.append(f"Executed script: {script_decoded[:50]}...")
            return {"message": "Script executed successfully"}
        except Exception as e:
            _action_history.append(f"Script execution failed: {str(e)}")
            raise Exception(f"Script execution failed: {str(e)}")

    def edit_mesh(self, object_name: str, operation: str, parameters: dict, verbose: bool = False, sequence: List[Dict[str, Any]] = None) -> dict:
        """Perform advanced mesh editing operations on the specified object, with optional sequence of operations."""
        try:
            obj = bpy.data.objects.get(object_name)
            if not obj or obj.type != 'MESH':
                return {"status": "error", "message": f"Object {object_name} not found or not a mesh", "suggestion": "Verify object name and ensure it’s a mesh"}

            initial_vertex_count = len(obj.data.vertices)
            bpy.context.view_layer.objects.active = obj
            original_mode = obj.mode
            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(obj.data)

            result = {"status": "success", "message": "", "affected_vertices": 0, "suggest_next_operation": None}

            # Handle sequence if provided
            if sequence:
                for step in sequence:
                    op = step.get("operation")
                    params = step.get("parameters", {})
                    if op in ["select_vertices", "extrude", "scale", "rotate", "move", "bevel", "smooth", "subdivide", "apply_curve"]:
                        result = self._perform_operation(bm, op, params, result, verbose)
                        if result["status"] == "error":
                            break
                    else:
                        result["status"] = "error"
                        result["message"] = f"Invalid operation in sequence: {op}"
                        break
                bmesh.update_edit_mesh(obj.data)
                if not parameters.get("keep_edit_mode", False):
                    bpy.ops.object.mode_set(mode=original_mode)
                return result

            # Handle single operation
            result = self._perform_operation(bm, operation, parameters, result, verbose)
            bmesh.update_edit_mesh(obj.data)
            if not parameters.get("keep_edit_mode", False):
                bpy.ops.object.mode_set(mode=original_mode)
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "suggestion": "Check object existence or try a simpler operation first"}

    def _perform_operation(self, bm, operation, parameters, result, verbose):
        """Helper method to perform a single mesh operation."""
        initial_vertex_count = len(bm.verts)
        if operation == "select_vertices":
            selection_mode = parameters.get("selection_mode", "replace")
            if selection_mode == "replace":
                for v in bm.verts:
                    v.select = False
            criteria = parameters.get("criteria", {})
            count = 0
            if "position_bounds" in criteria:
                bounds = criteria["position_bounds"]
                min_bounds = bounds.get("min", [-float('inf')] * 3)
                max_bounds = bounds.get("max", [float('inf')] * 3)
                for v in bm.verts:
                    if all(min_bounds[i] <= v.co[i] <= max_bounds[i] for i in range(3)):
                        v.select = selection_mode != "subtract"
                        count += 1
            elif "vertex_group" in criteria:
                group_name = criteria["vertex_group"]
                obj = bpy.context.object
                if group_name in obj.vertex_groups:
                    group_index = obj.vertex_groups[group_name].index
                    deform_layer = bm.verts.layers.deform.verify()
                    for v in bm.verts:
                        if group_index in v[deform_layer]:
                            v.select = selection_mode != "subtract"
                            count += 1
                else:
                    return {"status": "error", "message": f"Vertex group {group_name} not found", "suggestion": "Create or check the vertex group"}
            elif "index_range" in criteria:
                start = criteria["index_range"].get("start", 0)
                end = criteria["index_range"].get("end", len(bm.verts) - 1)
                for i, v in enumerate(bm.verts):
                    if start <= i <= end:
                        v.select = selection_mode != "subtract"
                        count += 1
            elif "face_index" in criteria:
                face_idx = criteria["face_index"]
                if 0 <= face_idx < len(bm.faces):
                    face = bm.faces[face_idx]
                    for v in face.verts:
                        v.select = selection_mode != "subtract"
                        count += 1
            result["message"] = f"Selected {count} vertices"
            result["affected_vertices"] = count
            result["suggest_next_operation"] = "extrude" if count > 0 else None

        elif operation == "extrude":
            selected_verts = [v for v in bm.verts if v.select]
            selected_edges = [e for e in bm.edges if e.select]
            selected_faces = [f for f in bm.faces if f.select]
            if not (selected_verts or selected_edges or selected_faces):
                return {"status": "error", "message": "Nothing selected to extrude", "suggestion": "Select vertices, edges, or faces first"}
            direction = parameters.get("direction", "normal")
            distance = parameters.get("distance", 1.0)
            if direction == "normal":
                if selected_faces:
                    extruded = bmesh.ops.extrude_face_region(bm, geom=selected_faces)
                elif selected_edges:
                    extruded = bmesh.ops.extrude_edge_only(bm, edges=selected_edges)
                else:
                    extruded = bmesh.ops.extrude_vert_indiv(bm, verts=selected_verts)
                verts_extruded = [v for v in extruded["geom"] if isinstance(v, bmesh.types.BMVert)]
                bmesh.ops.translate(bm, vec=Vector([0, 0, distance]), verts=verts_extruded)
            else:
                vec = Vector(direction) * distance
                if selected_faces:
                    extruded = bmesh.ops.extrude_face_region(bm, geom=selected_faces)
                elif selected_edges:
                    extruded = bmesh.ops.extrude_edge_only(bm, edges=selected_edges)
                else:
                    extruded = bmesh.ops.extrude_vert_indiv(bm, verts=selected_verts)
                verts_extruded = [v for v in extruded["geom"] if isinstance(v, bmesh.types.BMVert)]
                bmesh.ops.translate(bm, vec=vec, verts=verts_extruded)
            result["message"] = f"Extruded with direction {direction} and distance {distance}"
            result["affected_vertices"] = len(verts_extruded)
            result["suggest_next_operation"] = "scale" if len(verts_extruded) > 0 else None

        elif operation == "scale":
            selected_verts = [v for v in bm.verts if v.select]
            if not selected_verts:
                return {"status": "error", "message": "No vertices selected for scaling", "suggestion": "Select vertices first"}
            values = parameters.get("values", [1.0, 1.0, 1.0])
            pivot = parameters.get("pivot_point", "median")
            if pivot == "median":
                sum_co = Vector((0, 0, 0))
                for v in selected_verts:
                    sum_co += v.co
                pivot_co = sum_co / len(selected_verts)
            elif pivot == "cursor":
                pivot_co = Vector(bpy.context.scene.cursor.location)
            elif pivot == "center":
                pivot_co = bpy.context.object.location
            elif pivot == "individual":
                for v in selected_verts:
                    v.co.x *= values[0]
                    v.co.y *= values[1]
                    v.co.z *= values[2]
                result["message"] = f"Scaled {len(selected_verts)} vertices individually"
                result["affected_vertices"] = len(selected_verts)
                result["suggest_next_operation"] = "rotate"
            else:
                pivot_co = Vector(pivot)
            if pivot != "individual":
                for v in selected_verts:
                    delta = v.co - pivot_co
                    delta.x *= values[0]
                    delta.y *= values[1]
                    delta.z *= values[2]
                    v.co = pivot_co + delta
                result["message"] = f"Scaled {len(selected_verts)} vertices"
                result["affected_vertices"] = len(selected_verts)
                result["suggest_next_operation"] = "rotate"

        elif operation == "rotate":
            selected_verts = [v for v in bm.verts if v.select]
            if not selected_verts:
                return {"status": "error", "message": "No vertices selected for rotation", "suggestion": "Select vertices first"}
            angles = parameters.get("values", [0.0, 0.0, 0.0])
            pivot = parameters.get("pivot_point", "median")
            if pivot == "median":
                sum_co = Vector((0, 0, 0))
                for v in selected_verts:
                    sum_co += v.co
                pivot_co = sum_co / len(selected_verts)
            elif pivot == "cursor":
                pivot_co = Vector(bpy.context.scene.cursor.location)
            elif pivot == "center":
                pivot_co = bpy.context.object.location
            else:
                pivot_co = Vector(pivot)
            rot_x = Matrix.Rotation(angles[0], 4, 'X')
            rot_y = Matrix.Rotation(angles[1], 4, 'Y')
            rot_z = Matrix.Rotation(angles[2], 4, 'Z')
            rotation = rot_z @ rot_y @ rot_x
            for v in selected_verts:
                delta = v.co - pivot_co
                delta = rotation @ delta
                v.co = pivot_co + delta
            result["message"] = f"Rotated {len(selected_verts)} vertices"
            result["affected_vertices"] = len(selected_verts)
            result["suggest_next_operation"] = "smooth"

        elif operation == "move":
            selected_verts = [v for v in bm.verts if v.select]
            if not selected_verts:
                return {"status": "error", "message": "No vertices selected for movement", "suggestion": "Select vertices first"}
            values = Vector(parameters.get("values", [0.0, 0.0, 0.0]))
            bmesh.ops.translate(bm, vec=values, verts=selected_verts)
            result["message"] = f"Moved {len(selected_verts)} vertices"
            result["affected_vertices"] = len(selected_verts)
            result["suggest_next_operation"] = "extrude"

        elif operation == "bevel":
            selected_edges = [e for e in bm.edges if e.select]
            if not selected_edges:
                return {"status": "error", "message": "No edges selected for beveling", "suggestion": "Select edges first"}
            offset = parameters.get("offset", 0.1)
            segments = parameters.get("segments", 1)
            bmesh.ops.bevel(bm, geom=selected_edges, offset=offset, segments=segments, affect='EDGES')
            result["message"] = f"Beveled {len(selected_edges)} edges"
            result["affected_vertices"] = len(bm.verts) - initial_vertex_count
            result["suggest_next_operation"] = "smooth"

        elif operation == "smooth":
            selected_verts = [v for v in bm.verts if v.select]
            if not selected_verts:
                return {"status": "error", "message": "No vertices selected for smoothing", "suggestion": "Select vertices first"}
            factor = parameters.get("factor", 0.5)
            iterations = parameters.get("iterations", 1)
            for _ in range(iterations):
                bmesh.ops.smooth_vert(bm, verts=selected_verts, factor=factor)
            result["message"] = f"Smoothed {len(selected_verts)} vertices over {iterations} iterations"
            result["affected_vertices"] = len(selected_verts)
            result["suggest_next_operation"] = "subdivide"

        elif operation == "subdivide":
            selected_faces = [f for f in bm.faces if f.select]
            if not selected_faces:
                return {"status": "error", "message": "No faces selected for subdivision", "suggestion": "Select faces first"}
            cuts = parameters.get("cuts", 1)
            bmesh.ops.subdivide_edges(bm, edges=[e for e in bm.edges if e.select or any(f in selected_faces for f in e.link_faces)], cuts=cuts)
            result["message"] = f"Subdivided with {cuts} cuts"
            result["affected_vertices"] = len(bm.verts) - initial_vertex_count
            result["suggest_next_operation"] = "smooth"

        elif operation == "apply_curve":
            selected_verts = [v for v in bm.verts if v.select]
            if not selected_verts:
                return {"status": "error", "message": "No vertices selected for curve deformation", "suggestion": "Select vertices first"}
            curve_type = parameters.get("curve_type", "bezier")
            control_points = parameters.get("control_points", [])
            if len(control_points) < 2:
                return {"status": "error", "message": "At least 2 control points required", "suggestion": "Provide more control points"}
            bpy.ops.object.mode_set(mode='OBJECT')
            curve_data = bpy.data.curves.new('Curve', 'CURVE')
            curve_data.dimensions = '3D'
            spline = curve_data.splines.new(curve_type.upper())
            spline.points.add(len(control_points) - 1)
            for i, point in enumerate(control_points):
                spline.points[i].co = point + [1.0]  # Homogeneous coordinates
            curve_obj = bpy.data.objects.new('CurveObj', curve_data)
            bpy.context.scene.collection.objects.link(curve_obj)
            mod = bpy.context.object.modifiers.new(name="CurveMod", type='CURVE')
            mod.object = curve_obj
            bpy.context.view_layer.objects.active = bpy.context.object
            bpy.ops.object.modifier_apply(modifier="CurveMod")
            bpy.data.objects.remove(curve_obj)
            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(bpy.context.object.data)
            result["message"] = f"Applied {curve_type} curve to {len(selected_verts)} vertices"
            result["affected_vertices"] = len(selected_verts)
            result["suggest_next_operation"] = "smooth"

        if verbose:
            result["verbose_log"] = f"Performed {operation} on {result['affected_vertices']} vertices."

        return result

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