bl_info = {
    "name": "Toggle Selected Visibility",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Tool",
    "description": "Toggle viewport visibility of selected objects",
    "category": "Object",
}

import bpy

objects = ["2026-01-22_Kidney_1b", "2026-01-22_Kidney_1c"]


class OBJECT_OT_toggle_selected_visibility(bpy.types.Operator):
    """Toggle viewport visibility for selected objects"""

    bl_idname = "object.toggle_selected_visibility"
    bl_label = "Toggle Selected Visibility"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected_objects = context.selected_objects

        #        if not selected_objects:
        #            self.report({'WARNING'}, "No objects selected")
        #            return {'CANCELLED'}

        for obj_name in objects:
            obj = bpy.data.objects.get(obj_name)
            #            print(f"Hiding {obj.name}")
            #            obj.hide_viewport = not obj.hide_viewport
            print(f"Object visibility {obj.hide_get()} {obj.name}")
            obj.hide_set(not obj.hide_get())

        return {"FINISHED"}


class VIEW3D_PT_toggle_visibility_panel(bpy.types.Panel):
    """Creates a Panel in the 3D Viewport Sidebar"""

    bl_label = "Visibility Tools"
    bl_idname = "VIEW3D_PT_toggle_visibility_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tool"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.toggle_selected_visibility", icon="HIDE_OFF")


def register():
    bpy.utils.register_class(OBJECT_OT_toggle_selected_visibility)
    bpy.utils.register_class(VIEW3D_PT_toggle_visibility_panel)


def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_toggle_visibility_panel)
    bpy.utils.unregister_class(OBJECT_OT_toggle_selected_visibility)


if __name__ == "__main__":
    register()
