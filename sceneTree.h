#pragma once

#include <memory>
#include <vector>
#include <algorithm>

#include "bbox.h"
#include "object.h"
#include "intersect.h"
#include "light.h"

/*
  Un arbre contient soit des feuilles (des objets), soit des noeuds avec une boite et deux sous arbres.
*/

struct Leaf
{
	const Object *object;
};

struct Node;

using Tree = std::variant<Leaf, Node>;

struct Node
{
	BBox box;

	std::unique_ptr<Tree> childA;
	std::unique_ptr<Tree> childB;
};

std::ostream& operator<<(std::ostream &s, const Tree &t);

struct DisplayTree
{
	std::ostream &s;

	void operator()(const Leaf &l)
	{
		s << "Leaf{";
		s << *l.object;
		s << "}";
	}

	void operator()(const Node &l)
	{
		s << "Node{";
		s << *(l.childA);
		s << ",";
		s << *(l.childB);
		s << "}";
	}
};

std::ostream& operator<<(std::ostream &s, const Tree &t)
{
	std::visit(DisplayTree{s}, t);

	return s;
}

float getAxe(const Vec &v, const int axe)
{
	switch(axe)
	{
	case 0: return v.x;
	case 1: return v.y;
	case 2: return v.z;
	}

	return -10;
}

// boite engrobante d'un iterateur d'Object
BBox computeBox(decltype(std::vector<Object>{}.begin()) start, decltype(std::vector<Object>{}.end()) end)
{
	BBox box = empty();

	for(auto curIt = start; curIt != end; ++curIt)
	{
		box = bboxUnion(box, bboxSphere(curIt->sphere));
	}

	return box;
}

float cost(const BBox &V, const BBox &Vl, const BBox &Vr, const int Nl, const int Nr)
{
	const float surfaceV = surface(V);
	const float surfaceVl = surface(Vl);
	const float surfaceVr = surface(Vr);

	const float Kt = 1.f;
	const float Ki = 1.f;

	return Kt + Ki * (surfaceVl / surfaceV * Nl + surfaceVr / surfaceV * Nr);
}

// Construction récursive de larbre
std::unique_ptr<Tree> buildTree(decltype(std::vector<Object>{}.begin()) start, decltype(std::vector<Object>{}.end()) end)
{
	const auto size = std::distance(start, end);

	// feuille avec un seul element
	if(size == 1)
	{
		return std::make_unique<Tree>(Leaf{&*start});
	}

	// Compute the box of the objects
	const BBox box = computeBox(start, end);

	// split against the bigest axe
	auto boxSize = bboxSize(box);
	auto axe = (boxSize.z > boxSize.y && boxSize.z > boxSize.x ? 2 : (boxSize.y > boxSize.x ? 1 : 0));

	// on trie les objets selon l'axe
	std::sort(start, end, [axe](const Object &objectA, const Object &objectB)
			  {
				  return getAxe(objectA.sphere.center.value, axe) <
					  getAxe(objectB.sphere.center.value, axe);
			  });

	// et recursivement on fait l'abre gauche et droit avec la moitié des objets
	const auto middle = start + size / 2;
	return std::make_unique<Tree>(Node{box, buildTree(start, middle), buildTree(middle, end)});
}

std::optional<Intersection> intersectScene(const Ray &ray, const Tree &objects, const float minT);

struct IntersectVisitor
{
	const Ray &ray;
	const float minT;

	// Intersection d'une feuille -> intersection de la sphere
	std::optional<Intersection> operator()(const Leaf &l)
	{
		auto it = intersectSphere(ray, l.object->sphere);

		if(it && *it < minT)
		{
			return Intersection{*it, l.object};
		}

		return std::nullopt;
	}

	// Intersection d'un noeud
	std::optional<Intersection> operator()(const Node &n)
	{
		auto it = intersectBBox(ray, n.box);

		// on ne garde que les rayons qui commence dans la boite ou
		// qui intersectent la boite AVANT le tminimum connu
		if(pointInBox(ray.origin, n.box) || (it && *it < minT))
		{
			// calcul du l'intersection avec le premier fils
			auto itA = intersectScene(ray, *n.childA, minT);

			// calcul de l'intersection avec le second, mais avec un
			// tMin mis à jour connaissant l'intersection avec le
			// premier
			auto itB = intersectScene(ray, *n.childB, (itA ? std::min(itA->t, minT) : minT));

			if(itA && !itB)
			{
				return itA;
			}
			else if(itB && !itA)
			{
				return itB;
			}
			else
			{
				// intersection dans les deux, on renvoi la plus proche
				if(itA->t < itB->t)
				{
					return itA;
				}
				return itB;
			}
		}

		return std::nullopt;
	}
};

std::optional<Intersection> intersectScene(const Ray &ray, const Tree &objects)
{
	// le tMin actuel c'est l'infini
	return intersectScene(ray, objects, std::numeric_limits<float>::infinity());
}

std::optional<Intersection> intersectScene(const Ray &ray, const Tree &objects, const float minT)
{
	return std::visit(IntersectVisitor{ray, minT}, objects);
}

struct Scene
{
	std::vector<Object> objects;
	std::vector<Light> lights;

	std::unique_ptr<const Tree> tree;

	Scene(const std::vector<Object> &_objects, const std::vector<Light> &_lights)
		:objects(_objects), lights(_lights)
	{
		tree = buildTree(std::begin(objects), std::end(objects));
	}
};
