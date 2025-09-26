#!/usr/bin/env python3
"""
PyEidors综合测试套件
运行所有可用的测试，包括基础模块测试、功能测试和性能测试
"""

import subprocess
import sys
import os
from pathlib import Path
import time


class TestRunner:
    """测试运行器类"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    def run_test(self, test_name: str, test_script: str) -> bool:
        """
        运行单个测试脚本
        
        参数:
            test_name: 测试名称
            test_script: 测试脚本路径
            
        返回:
            测试是否成功
        """
        print(f"\n{'='*60}")
        print(f"正在运行: {test_name}")
        print(f"脚本: {test_script}")
        print(f"{'='*60}")
        
        if not Path(test_script).exists():
            print(f"❌ 测试脚本不存在: {test_script}")
            self.results[test_name] = {'status': 'missing', 'time': 0}
            return False
        
        start_time = time.time()
        
        try:
            # 运行测试脚本
            result = subprocess.run(
                [sys.executable, test_script],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            test_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {test_name} 测试通过")
                self.results[test_name] = {'status': 'passed', 'time': test_time}
                
                # 如果有输出，显示最后几行
                if result.stdout:
                    output_lines = result.stdout.strip().split('\n')
                    print("📋 测试输出摘要:")
                    for line in output_lines[-5:]:  # 显示最后5行
                        print(f"   {line}")
                
                return True
            else:
                print(f"❌ {test_name} 测试失败 (退出码: {result.returncode})")
                self.results[test_name] = {'status': 'failed', 'time': test_time}
                
                # 显示错误信息
                if result.stderr:
                    print("🔍 错误信息:")
                    error_lines = result.stderr.strip().split('\n')
                    for line in error_lines[-10:]:  # 显示最后10行错误
                        print(f"   {line}")
                
                return False
                
        except subprocess.TimeoutExpired:
            test_time = time.time() - start_time
            print(f"⏰ {test_name} 测试超时")
            self.results[test_name] = {'status': 'timeout', 'time': test_time}
            return False
            
        except Exception as e:
            test_time = time.time() - start_time
            print(f"💥 {test_name} 测试执行异常: {e}")
            self.results[test_name] = {'status': 'error', 'time': test_time}
            return False
    
    def print_summary(self):
        """打印测试总结"""
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print("🏆 PyEidors测试套件运行完成")
        print(f"{'='*80}")
        
        print(f"\n📊 测试结果统计:")
        print(f"{'测试名称':<30} {'状态':<10} {'时间(秒)':<10}")
        print(f"{'-'*55}")
        
        passed = failed = timeout = error = missing = 0
        
        for test_name, result in self.results.items():
            status = result['status']
            test_time = result['time']
            
            if status == 'passed':
                status_emoji = "✅ 通过"
                passed += 1
            elif status == 'failed':
                status_emoji = "❌ 失败"
                failed += 1
            elif status == 'timeout':
                status_emoji = "⏰ 超时"
                timeout += 1
            elif status == 'error':
                status_emoji = "💥 异常"
                error += 1
            else:
                status_emoji = "❓ 缺失"
                missing += 1
            
            print(f"{test_name:<30} {status_emoji:<10} {test_time:<10.2f}")
        
        total_tests = len(self.results)
        
        print(f"\n📈 总体统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed}")
        print(f"   失败: {failed}")
        print(f"   超时: {timeout}")
        print(f"   异常: {error}")
        print(f"   缺失: {missing}")
        print(f"   成功率: {passed/total_tests*100:.1f}%" if total_tests > 0 else "   成功率: 0%")
        print(f"   总用时: {total_time:.2f} 秒")
        
        # 提供建议
        print(f"\n💡 建议:")
        if failed > 0:
            print("   - 检查失败的测试，可能需要修复依赖或配置问题")
        if timeout > 0:
            print("   - 超时的测试可能需要优化性能或增加超时时间")
        if error > 0:
            print("   - 出现异常的测试需要检查代码错误")
        if missing > 0:
            print("   - 缺失的测试脚本需要创建")
        if passed == total_tests:
            print("   - 🎉 所有测试都通过了！系统运行良好。")
        
        return passed == total_tests


def main():
    """主函数"""
    print("🚀 启动PyEidors综合测试套件")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    runner = TestRunner()
    
    # 定义测试列表
    tests = [
        ("基础模块测试", "test_pyeidors.py"),
        ("简化系统测试", "test_simplified_eit_system.py"),
        ("完整系统测试", "test_complete_eit_system.py"),
    ]
    
    all_passed = True
    
    # 运行所有测试
    for test_name, test_script in tests:
        success = runner.run_test(test_name, test_script)
        if not success:
            all_passed = False
    
    # 打印总结
    runner.print_summary()
    
    # 创建测试报告
    create_test_report(runner.results)
    
    return all_passed


def create_test_report(results):
    """创建详细的测试报告"""
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "test_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# PyEidors测试报告\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 测试结果\n\n")
        f.write("| 测试名称 | 状态 | 时间(秒) |\n")
        f.write("|----------|------|----------|\n")
        
        for test_name, result in results.items():
            status = result['status']
            test_time = result['time']
            
            status_map = {
                'passed': '✅ 通过',
                'failed': '❌ 失败', 
                'timeout': '⏰ 超时',
                'error': '💥 异常',
                'missing': '❓ 缺失'
            }
            
            status_text = status_map.get(status, status)
            f.write(f"| {test_name} | {status_text} | {test_time:.2f} |\n")
        
        f.write("\n## 系统信息\n\n")
        f.write(f"- Python版本: {sys.version}\n")
        f.write(f"- 平台: {sys.platform}\n")
        f.write(f"- 工作目录: {os.getcwd()}\n")
        
        f.write("\n## 模块状态\n\n")
        try:
            import pyeidors
            env = pyeidors.check_environment()
            f.write(f"- FEniCS: {'✅' if env['fenics_available'] else '❌'}\n")
            f.write(f"- PyTorch: {'✅' if env['torch_available'] else '❌'}\n")
            f.write(f"- CUDA: {'✅' if env['cuda_available'] else '❌'}\n")
            f.write(f"- CUQIpy: {'✅' if env['cuqi_available'] else '❌'}\n")
            if env['torch_available']:
                f.write(f"- PyTorch版本: {env['torch_version']}\n")
                f.write(f"- GPU数量: {env['cuda_device_count']}\n")
        except Exception as e:
            f.write(f"- 环境检查失败: {e}\n")
    
    print(f"\n📄 详细测试报告已保存到: {report_file.absolute()}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)