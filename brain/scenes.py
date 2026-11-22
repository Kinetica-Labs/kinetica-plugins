"""The examples of the brain (plugin.py offers each as a button; the tests build them with no window): made by the
editor's operations, the same in the window and in a kinetica-core world."""


def balance(scene):
    """A cart and a metre-long pole on a free hinge: the brain keeps the pole up by the cart's thrust along x. It feels
    the pole's tilt by the hinge's encoder and how fast it falls by an IMU's gyroscope on its top (a balancing robot's
    way), and where the cart is by a lidar's one beam to the wall behind; a push on the pole at the start tips it.
    Returns the cart's id - the body the brain goes on."""
    ed = scene.ed
    cart = ed.scene_add(shape="box", name="Тележка", position=[0, 0.1, 0], size=[0.4, 0.2, 0.3], role="rigid")["id"]
    ed.scene_set(id=cart, line="rigid", key="mass", values=[2])
    ed.scene_set(id=cart, line="rigid", key="friction", values=[0.05])
    pole = ed.scene_add(shape="box", name="Шест", position=[0, 0.7, 0], size=[0.04, 1.0, 0.04], role="rigid")["id"]
    ed.scene_set(id=pole, line="rigid", key="mass", values=[0.3])
    joint = ed.assembly_joint(a=cart, b=pole, type="hinge", anchor=[0, 0.2, 0], axis=[0, 0, 1])["id"]
    ed.assembly_encoder(joint=joint, lines=1000, reference=0)
    ed.assembly_sensor(target=pole, kind="imu", face="+y")
    ed.scene_add(shape="box", name="Стена", position=[-2.5, 0.5, 0], size=[0.2, 1, 3], role="collider")
    ed.assembly_sensor(target=cart, kind="lidar", face="-x", model="TFmini-S")
    thrust = ed.scene_add_force(shape="directional", mode="constant", body=cart, strength=0)["id"]
    ed.scene_turn(ids=[thrust], axis=[0, 0, 1], degrees=-90)  # its +Y along the world's +x
    push = ed.scene_add_force(shape="directional", mode="impulse", body=pole, strength=0.06, at=[0, 1.1, 0])["id"]
    ed.scene_turn(ids=[push], axis=[0, 0, 1], degrees=-90)
    return cart
