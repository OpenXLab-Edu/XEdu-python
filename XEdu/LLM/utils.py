import socket
import ipaddress
def get_local_ip():
    """
    获取本机IP地址
    """
    try:
        # 创建一个socket对象
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 不需要连接到一个特定的地址，只需要一个有效的IP即可
        s.connect(('8.8.8.8', 80))
        # 获取本地IP地址
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        print(f"获取本机IP时发生错误: {e}")
        return None

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))  # 尝试绑定到指定端口
        except socket.error as e:
            if e.errno == 98:  # 错误号98通常表示"Address already in use"
                return True  # 端口被占用
            else:
                raise  # 其他错误重新抛出
        else:
            return False  # 绑定成功，端口未被占用

def find_available_port(start_port=7860):
    """从指定端口开始寻找下一个可用端口"""
    while is_port_in_use(start_port):
        start_port += 1
    return start_port

def is_public_ipv4(ip):
    """
    判断一个IPv4地址是否为公网地址。
    
    :param ip: 字符串形式的IPv4地址。
    :return: 布尔值，True表示是公网地址，False表示是私有地址或其他无效地址。
    """
    try:
        # 将IP地址转换为IPv4Address对象
        ip_obj = ipaddress.IPv4Address(ip)
        
        # 判断是否为私有IP
        if ip_obj.is_private:
            return False
        else:
            return True
    except ValueError:
        # 如果ip不是一个有效的IPv4地址，则返回False
        return False