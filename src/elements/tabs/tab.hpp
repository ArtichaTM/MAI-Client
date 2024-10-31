#pragma once
#include <vector>

#include <SFML/Graphics.hpp>

#include "../base.hpp"
#include "../button.hpp"

class Tab : public SFBase {
    std::vector<SFBase*> elements;
public:
    Button tabText;
    Tab(const std::string& title, float offset, float height, sf::Color color);
    ~Tab();
    void draw(sf::RenderWindow& window) const override;
    void draw(sf::RenderWindow& window, bool active = false) const;
    void handleEvent(const sf::Event& event) override;
    bool IsSwitcherInBounds(const sf::Vector2i& position) const;
    bool IsSwitcherInBounds(const sf::Vector2f& position) const;
    void AddElement(SFBase* element);
    const std::string getName();
};
