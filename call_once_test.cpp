#include <iostream>
//#include <thread>
#include <mutex>
#include <memory>
#include <pthread.h>

std::once_flag flag1, flag2;

void 
DoSomething() 
{
	std::call_once(flag1, []() {std::cout << "hello world!" << std::endl;});
}

void 
may_throw_function(bool do_throw)
{
	if (do_throw) {
		std::cout << "throw: call_once will retry \n";
		throw std::exception();	
	}
	std::cout << "not throw \n";
}


void
do_once(bool do_throw)
{
	try {
		std::call_once(flag2, may_throw_function, do_throw);	
	} catch(...) {}
}

int
main() 
{
	std::thread thread1(DoSomething);
	std::thread thread2(DoSomething);
	std::thread thread3(DoSomething);
	std::thread thread4(DoSomething);

	thread1.join();
	thread2.join();
	thread3.join();
	thread4.join();

	std::thread t1(do_once, true);
	std::thread t2(do_once, true);
	std::thread t3(do_once, false);
	std::thread t4(do_once, true);

	t1.join();
	t2.join();
	t3.join();
	t4.join();

	return 0;
}
