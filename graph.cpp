#include <memroy>
#include <vector>
#include <iostream>

template<typename T>
class Node
{
public:
	T m_value;
	std::vector<std::weak_ptr<Node<T>>> neighbours;

	Node (T value): m_value(value), neighbours{} 
	{
		std::cout << "Node(" << value << ")" << std::endl;
	}
	~Node()
	{
		std::cout << "~Node(" << value << ")" << std::endl;
	}
}

template<typename T>
class Graph: public std::vector<std::shared_ptr<Node<T>>>
{
public:
	std::weak_ptr<Node<T>> get(std::size_t i) 
	{
		auto ret = std::weak_ptr<Node<T>> {};
		ret = std::vector<std::shared_ptr<Node<T>>>::operator[](i);
		return ret;
	}
}

int main(int argc, char *argv[])
{
	Graph<int> g;
	g.emplace_back(std::make_shared<Node<T>>(0));
	g.emplace_back(std::make_shared<Node<T>>(1));

	// set the edges of the graph
	g[0]->neighbours.emplace_back(g.get(1));
	g[1]->neighbours.emplace_back(g.get(0));
	
	// start in the graph
	auto ptr = g.get(0);
	
	// move around in the graph, and print position
	std::cout << ptr.lock()->value << std::endl;
	ptr = ptr.lock()->neighbours[0];
	std::cout << ptr.lock()->value << std::endl;
	ptr = ptr.lock()->neighbours[0];
	std::cout << ptr.lock()->value << std::endl;
	ptr = ptr.lock()->neighbours[0];
 	std::cout << ptr.lock()->value << std::endl;
    	ptr = ptr.lock()->neighbours[0];
    	std::cout << ptr.lock()->value << std::endl;
    	ptr = ptr.lock()->neighbours[0];
    	std::cout << ptr.lock()->value << std::endl;

	return 0;
}

































