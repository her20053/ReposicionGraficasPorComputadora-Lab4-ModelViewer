class Obj(object):
  def __init__(self, filename):

    with open(filename) as f:
      self.lines = f.read().splitlines()

    self.vertices = []
    self.faces = []
    self.tvertices = []
    self.nvertices = []

    for line in self.lines:

      if line:

        if ' ' not in line:
          continue

        prefix, value = line.split(' ', 1)

        if value[0] == ' ':
          value = '' + value[1:]
        
        if prefix == 'v':
          self.vertices.append(
            list(
              map(float, value.split(' '))
            )
          )
        
        if prefix == 'vt':
          self.tvertices.append(
            list(
              map(float, value.split(' '))
            )
          )

        if prefix == 'vn':
          self.nvertices.append(
            list(
              map(float, value.split(' '))
            )
          )

        if prefix == 'f':
          temp = value.replace('//', '/0/')
          act_face = [
            list(map(int, face.split('/')))
                for face in temp.split(' ') if face != ''
          ]
          self.faces.append(act_face) 