# Blender MCP VXAI 

Pro Tip: You can use this to ask your agent to export the 3d model directly into the project you are working on to use them in your app instantly. [Demo](https://youtu.be/sHRI0nPan20?feature=shared)
## Description

Blender MCP VXAI a powerful integration that allows you to control Blender using natural language through MCP Clients. This tool enables you to create, modify, and manipulate 3D models, animations, and scenes in Blender by simply describing what you want to do. It bridges the gap between AI language models and 3D creation, making Blender more accessible and efficient for both beginners and experienced users. This is a simple tool where the agent can let the AI agent create scripts for you while getting feedback and building the exact scene you want.

## v1.0.3 Released PLEASE REPEAT THE ADDON STEP WITH NEW ADDON FILE and get latest from uv for server. Please reach out if any issues.
Check out release notes for more details about enhancements [Demo](https://youtu.be/3e3h6rN194I?si=E7cuDKhsHK0mcRsO)


## Guide to Using the Latest Version Once You're Set Up

Follow these steps to create and explore your stunning 3D world!

### Step 1: Get a Basic Image
Start with a simple image that will serve as the foundation for your project.

### Step 2: Load It Into Your Tool
Upload your image into **Cursor**, **Cline**, **Windsurf**, or any **MCP client** of your choice.

### Step 3: Add a Creative Prompt
Enter a prompt to transform your image into something extraordinary. Try something like:

```plaintext
Create this in 3D. I've given you my insane architectural plans—make it as pretty as you can! :)
```

### Step 4: Tweak It with Natural Language
Refine your creation by making adjustments in plain English. Keep tweaking until your world looks just right.

### Step 5: Export and Build Your Interactive World
When you're satisfied, use this prompt to take it to the next level:

```plaintext
Export this scene in this project in .gib format, then create a ThreeJS app and use this as my world. Set it up as a server to avoid file-loading issues. I want to roam around this world freely—go wild!
```

### Step 6: Enjoy Your Creation
Take a moment to explore and admire your fully realized 3D world.

### Step 7: Make It Even Prettier
Polish it up with final touches to enhance its beauty—let your imagination shine!

---

Enjoy crafting and roaming your masterpiece!



## Features

- Control Blender using natural language commands
- Seamless integration with MCP Clients
- Automate complex Blender operations with simple text instructions
- Enhance your 3D workflow with AI assistance

## Installation

### Prerequisites

- Blender 
- Python 3.8+

### Step 1: Install UV

UV is required to run the MCP server.

**macOS:**
```bash
brew install uv
```

**Windows/Linux:**
```bash
pip install uv
```

### Step 2: Configure Your Environment
**For the latest version just use uvx blender-mcp-vxai**
#### For Cursor:
1. Click "+ Add new Server"
2. Configure with:
   - Name: `blender-mcp`
   - Command: `uvx blender-mcp-vxai==1.0.5`

#### For Claude Desktop:
1. Go to Claude > Settings > Developer > Edit Config
2. Open `claude_desktop_config.json` and add:
```json
{
    "mcpServers": {
        "blender": {
            "command": "uvx",
            "args": [
                "blender-mcp-vxai==1.0.5"
            ]
        }
    }
}
```

### Step 3: Install the Blender Addon

1. Download `blender_mcp_addon.py`
2. Open Blender
3. Go to Edit > Preferences > Add-ons
4. Click "Install from Disk" and select the `blender_mcp_addon.py` file
5. Enable the addon by checking the box next to "Blender MCP"
6. In Blender, go to the 3D View sidebar (press N if not visible)
7. Find the "BlenderMCP" tab
8. Start MCP server



## Usage


## Available Tools

- **Object Creation**: Create primitives, import models, and generate complex shapes
- **Modeling**: Modify meshes, add modifiers, and sculpt objects
- **Materials**: Create and apply materials, textures, and shaders
- **Animation**: Create keyframes, animate properties, and set up rigging
- **Rendering**: Configure render settings, lighting, and camera positions
- **Scene Management**: Organize objects, collections, and scenes

## Example Usecases
- Share an image and ask it to create a low poly version of it
- Update elements on scene as needed, describe it in detail
- Build simple to complex scenes one step a time
- change camera, color, add lighting etc


## Troubleshooting

- If the connection fails, ensure the MCP server is running
- Check that the addon is properly installed and enabled
- For specific errors, check the Blender console for detailed messages

### For older versions the command is

**uvx --from blender-mcp-vxai start**



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE

