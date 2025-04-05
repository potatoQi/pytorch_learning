'''
总结传参的方式:

如果只是普通传几个参数, 用 parser 就够了, 用的时候直接 args.xxx

如果想通过 yaml 配置文件对类进行实例化, 那么就 parser (读取 yaml 路径) + omegaconf (转为 omegaconf 对象) + hydra (实例化类)

    这种方式还有一个好处就是配置文件可以写多个, 合并配置文件的过程中, 如果后面的配置中有同名的键，会覆盖前面的配置值
    所以可以弄一个 config 文件夹, 然后不同的配置写到不同的 yaml 文件里, 保证清晰易读
'''

import argparse
import yaml                                         # pip install yaml
from omegaconf import OmegaConf                     # pip install omegaconf
from hydra.utils import instantiate                 # pip install hydra-core

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument('--int_num', type=int, default=1, help='int number', required=True)
    parser.add_argument('--float_num', type=float, default=0.1, help='float number', required=False)
    parser.add_argument('--str', type=str, default='hello', help='string')      # 如果字符串有空格要用引号括起来
    parser.add_argument('--is_ok', action='store_true', help='is ok')
    
    # 读多个值
    parser.add_argument('--base', '-b',
                        nargs='*',      # nargs='*' 表示接受0个或多个参数, 所以会返回一个 list
                        help='paths to base configs. Loaded from left-to-right. ', 
                        default=[],
                        required=True
                        )
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # 通过 args.xxx 获取命令行参数
    print('通过 args.xxx 获取命令行参数:')
    print(args.int_num)
    print(args.float_num)
    print(args.str)
    print(args.is_ok)
    print(args.base)
    print()

    # 把 yaml 文件读取成 字符串, 字典
    with open(args.base[0], 'r', encoding='utf-8') as f: conf_str = f.read()
    conf = yaml.safe_load(conf_str)
    print('字符串形式:\n', conf_str)     # 字符串形式, 能完整的体现出 yaml 文件的格式
    print('dict 形式:\n', conf)         # dict 格式
    print('version: ', conf['Students']['Normal_Students']['version'])
    print()

    # 通过 OmegaConf.load() 加载 yaml 文件
    configs = [OmegaConf.load(conf_path) for conf_path in args.base]
    # 通过 OmegaConf.merge() 合并多个配置文件, 在合并过程中，如果后面的配置中有同名的键，会覆盖前面的配置值。
    config = OmegaConf.merge(*configs)

    print('version: ', config.Students.Normal_Students.version)  # omega 对象可以通过 . 而非 dict 语法来访问
    print()

    # 通过 omegaconf 对象来实例化类
    normal_student = instantiate(config.Students.Normal_Students)
    normal_student.output()