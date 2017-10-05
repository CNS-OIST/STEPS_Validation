import os.path as osp


class Configuration(object):
    def __init__(self):
        self.mpi = False

    @property
    def meshes_dir(self):
        this_dir = osp.dirname(osp.abspath(__file__))
        validation_dir = 'validation_rd'
        if self.mpi:
            validation_dir += '_mpi'
        return osp.join(this_dir, validation_dir, 'meshes')

    def mesh_path(self, file):
        return osp.join(self.meshes_dir, file)

    def path(self, file):
        this_dir = osp.dirname(osp.abspath(__file__))
        return osp.join(this_dir, file)


configuration = Configuration()
del Configuration
