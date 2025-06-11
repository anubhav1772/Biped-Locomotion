from urdfpy import URDF
from graphviz import Digraph

robot = URDF.load('urdf/biped_flatfoot_robot.urdf')

dot = Digraph(comment='Kinematic Chain')

# Add nodes for links
for link in robot.links:
    dot.node(link.name)

# Add edges for joints (parent -> child)
for joint in robot.joints:
    dot.edge(joint.parent, joint.child, label=joint.name)

dot.render('kinematic_chain', format='png')