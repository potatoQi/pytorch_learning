class Normal_Student:
    def __init__(self,
                 version,
                 name,
                 instruction,
                 bool,
                 id,
                 ):
        self.version = version
        self.name = name
        self.instruction = instruction
        self.bool = bool
        self.id = id
    
    def output(self):
        print(f'version: {self.version}')
        print(f'name: {self.name}')
        print(f'instruction: {self.instruction}')
        print(f'bool: {self.bool}')
        print(f'id: {self.id}')