import subprocess

def check_tmp_filesystem():
    """Check if /tmp is a memory filesystem."""
    try:
        result = subprocess.run(['df', '-T', '/tmp'],
                              capture_output=True, text=True)
        if 'tmpfs' in result.stdout:
            print("/tmp uses tmpfs (memory filesystem) - extremely fast IO performance!")
        else:
            print("/tmp uses regular disk filesystem")
        print(result.stdout)
    except:
        pass

check_tmp_filesystem()
