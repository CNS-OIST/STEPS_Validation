import os
import os.path as osp


class Configuration(object):
    def __init__(self):
        self.suffix = ''

    @property
    def meshes_dir(self):
        """Get absolute path to the `validation_rd/meshes` directory
        """
        this_dir = osp.dirname(osp.abspath(__file__))
        validation_dir = 'validation_rd' + self.suffix
        return osp.join(this_dir, validation_dir, 'meshes')

    def mesh_path(self, file):
        """Get absolute path to the mesh file given in parameter
        """
        return osp.join(self.meshes_dir, file)

    def path(self, file):
        """Get absolute path from a path relative to the `./validation` directory
        """
        this_dir = osp.dirname(osp.abspath(__file__))
        return osp.join(this_dir, file)

    def checkpoint(self, name):
        """Get path to checkpoint given in parameter
        """
        checkpoints_dir = osp.join(os.getcwd(), 'checkpoints')
        if not osp.isdir(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        return osp.join(checkpoints_dir, name)


configuration = Configuration()
del Configuration
