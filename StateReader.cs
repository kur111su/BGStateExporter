using System;
using System.Collections.Generic;
using System.Linq;
using HearthDb.Enums;
using Hearthstone_Deck_Tracker;
using Hearthstone_Deck_Tracker.Hearthstone;
using Hearthstone_Deck_Tracker.Hearthstone.Entities;

namespace BGStateExporter
{
	/// <summary>
	/// 从 HDT 的 Core.Game 读取当前战旗局面，转换为 BgState。
	/// 必须在 HDT 的 UI 线程（OnUpdate）里调用。
	/// </summary>
	internal static class StateReader
	{
		public static BgState Read()
		{
			var game = Core.Game;
			var state = new BgState();

			if(game.IsInMenu || !game.IsRunning)
			{
				state.Phase = "menu";
				return state;
			}

			state.InGame = true;
			state.IsBattlegrounds = game.IsBattlegroundsMatch;
			if(!state.IsBattlegrounds)
			{
				state.Phase = "not_battlegrounds";
				return state;
			}

			state.Mode = game.IsBattlegroundsDuosMatch ? "duos" : "solo";
			state.Turn = game.GetTurnNumber();

			var races = BattlegroundsUtils.GetAvailableRaces();
			if(races != null)
				state.AvailableRaces = races.Select(r => r.ToString()).OrderBy(r => r).ToList();
			state.AnomalyDbfId = BattlegroundsUtils.GetBattlegroundsAnomalyDbfId(game.GameEntity);

			// 复制一份实体列表，避免日志线程同时修改字典时枚举出错
			var entities = game.Entities.Values.ToList();

			var playerHero = game.Player.Hero;
			if(state.Turn == 0)
			{
				state.Phase = "hero_select";
				state.Leaderboard = ReadLeaderboard(entities, game.Player.Id);
				return state;
			}
			if(playerHero == null)
			{
				// 对局中英雄实体会短暂为空（变身、换英雄技能时），这一帧只给阶段，不给玩家数据
				state.Phase = game.IsBattlegroundsCombatPhase ? "combat" : "recruit";
				state.Leaderboard = ReadLeaderboard(entities, game.Player.Id);
				return state;
			}

			state.NextOpponentPlayerId = game.PlayerEntity?.GetTag(GameTag.NEXT_OPPONENT_PLAYER_ID) ?? 0;
			state.Phase = game.IsBattlegroundsCombatPhase ? "combat" : "recruit";
			if(playerHero.Health <= 0)
				state.Phase = "game_over";

			state.Player = ReadPlayer(game, playerHero);

			var oppBoard = SortedBoard(game.Opponent.Board.Where(e => e.IsMinion || e.IsBattlegroundsSpell));
			if(state.Phase == "recruit")
			{
				// 招募阶段，对面的“玩家”是鲍勃，他场上的随从就是酒馆
				state.Tavern = oppBoard.Select(ToCard).ToList();
			}
			else if(state.Phase == "combat")
			{
				var oppHero = game.Opponent.Hero;
				state.Opponent = new OpponentState
				{
					HeroCardId = oppHero?.CardId,
					HeroName = oppHero?.LocalizedName,
					Health = oppHero?.Health ?? 0,
					Armor = oppHero?.GetTag(GameTag.ARMOR) ?? 0,
					TavernTier = oppHero?.GetTag(GameTag.PLAYER_TECH_LEVEL) ?? 0,
					Board = oppBoard.Where(e => e.IsMinion).Select(ToCard).ToList(),
				};
			}

			state.Leaderboard = ReadLeaderboard(entities, game.Player.Id);
			return state;
		}

		private static PlayerState ReadPlayer(GameV2 game, Entity hero)
		{
			var pe = game.PlayerEntity;
			var resources = pe?.GetTag(GameTag.RESOURCES) ?? 0;
			var temp = pe?.GetTag(GameTag.TEMP_RESOURCES) ?? 0;
			var used = pe?.GetTag(GameTag.RESOURCES_USED) ?? 0;
			var board = game.Player.Board.ToList();

			return new PlayerState
			{
				HeroCardId = hero.CardId,
				HeroName = hero.LocalizedName,
				Health = hero.Health,
				Armor = hero.GetTag(GameTag.ARMOR),
				TavernTier = hero.GetTag(GameTag.PLAYER_TECH_LEVEL),
				Gold = Math.Max(0, resources + temp - used),
				MaxGold = resources,
				ExtraGoldNextTurn = Math.Max(pe?.GetTag(GameTag.BACON_PLAYER_EXTRA_GOLD_NEXT_TURN) ?? 0,
				                             hero.GetTag(GameTag.BACON_PLAYER_EXTRA_GOLD_NEXT_TURN)),
				LeaderboardPlace = hero.GetTag(GameTag.PLAYER_LEADERBOARD_PLACE),
				HeroPowers = board.Where(e => e.IsHeroPower).Select(hp => new HeroPowerState
				{
					EntityId = hp.Id,
					CardId = hp.CardId,
					Name = hp.LocalizedName,
					Cost = hp.Cost,
					Used = hp.HasTag(GameTag.EXHAUSTED),
				}).ToList(),
				Board = SortedBoard(board.Where(e => e.IsMinion)).Select(ToCard).ToList(),
				Hand = game.Player.Hand.OrderBy(e => e.ZonePosition).Select(ToCard).ToList(),
				Trinkets = game.Player.Trinkets.Select(ToCard).ToList(),
				QuestRewards = game.Player.QuestRewards.Select(ToCard).ToList(),
				Shop = ReadShop(game, hero.GetTag(GameTag.PLAYER_TECH_LEVEL)),
			};
		}

		/// <summary>
		/// 读取商店各按钮的费用。战旗里随从身上的 COST 一般是 0，真正的价格写在
		/// “拖动购买 / 刷新 / 升级酒馆 / 冻结”这几个按钮实体上。找不到时用默认值，并标记为未知。
		/// </summary>
		internal static ShopInfo ReadShop(GameV2 game, int tier)
		{
			var ents = game.Player.PlayerEntities.Where(e => e.CardId != null).ToList();
			Entity? Find(Func<string, bool> match) =>
				ents.Where(e => match(e.CardId!)).OrderByDescending(e => e.Id).FirstOrDefault();

			var shop = new ShopInfo();
			var buy = Find(id => id == "TB_BaconShop_DragBuy");
			if(buy != null && buy.HasTag(GameTag.COST)) { shop.BuyCost = buy.GetTag(GameTag.COST); shop.BuyCostKnown = true; }
			var spellBuy = Find(id => id == "TB_BaconShop_DragBuy_Spell");
			if(spellBuy != null) shop.SpellBuyCost = spellBuy.GetTag(GameTag.COST);
			var reroll = Find(id => id == "TB_BaconShop_8p_Reroll_Button");
			if(reroll != null) { shop.RefreshCost = reroll.GetTag(GameTag.COST); shop.RefreshCostKnown = true; }
			var freeze = Find(id => id == "TB_BaconShopLockAll_Button");
			if(freeze != null) shop.FreezeCost = freeze.GetTag(GameTag.COST);

			// 升级按钮的卡牌 ID 带着目标等级：TB_BaconShopTechUp0N_Button，N = 当前等级 + 1
			var nextTier = tier + 1;
			var techUp = Find(id => id == $"TB_BaconShopTechUp0{nextTier}_Button");
			if(techUp != null)
			{
				shop.UpgradeCost = techUp.GetTag(GameTag.COST);
				shop.CanUpgrade = true;
			}
			shop.FreeRefreshes = game.PlayerEntity?.GetTag(GameTag.BACON_NUMBER_HERO_REFRESH_AVAILABLE) ?? 0;
			return shop;
		}

		private static List<LeaderboardEntry> ReadLeaderboard(List<Entity> entities, int playerId)
		{
			// 每个名次可能有多个英雄实体（战斗时会复制），取 Id 最大的那个
			return entities
				.Where(e => e.IsHero && e.GetTag(GameTag.PLAYER_LEADERBOARD_PLACE) > 0)
				.GroupBy(e => e.GetTag(GameTag.PLAYER_LEADERBOARD_PLACE))
				.Select(g => g.OrderByDescending(e => e.Id).First())
				.OrderBy(e => e.GetTag(GameTag.PLAYER_LEADERBOARD_PLACE))
				.Select(e => new LeaderboardEntry
				{
					Place = e.GetTag(GameTag.PLAYER_LEADERBOARD_PLACE),
					PlayerId = e.GetTag(GameTag.PLAYER_ID),
					HeroCardId = e.CardId,
					HeroName = e.LocalizedName,
					Health = e.Health,
					Armor = e.GetTag(GameTag.ARMOR),
					TavernTier = e.GetTag(GameTag.PLAYER_TECH_LEVEL),
					IsPlayer = e.IsControlledBy(playerId),
					Dead = e.Health <= 0,
				})
				.ToList();
		}

		private static IEnumerable<Entity> SortedBoard(IEnumerable<Entity> board) => board.OrderBy(e => e.ZonePosition);

		private static CardState ToCard(Entity e)
		{
			var card = e.Card;
			var races = new List<string>();
			AddRace(races, card.RaceEnum);
			AddRace(races, card.SecondaryRaceEnum);

			var tier = e.GetTag(GameTag.TECH_LEVEL);
			if(tier == 0)
				tier = card.TechLevel;

			return new CardState
			{
				EntityId = e.Id,
				CardId = e.CardId,
				DbfId = card.DbfId,
				Name = e.LocalizedName ?? e.Name,
				Type = e.IsMinion ? "minion" : (e.IsSpell || e.IsBattlegroundsSpell) ? "spell" : "other",
				Position = e.ZonePosition,
				Attack = e.Attack,
				Health = e.Health,
				Tier = tier,
				Cost = e.Cost,
				Races = races,
				Golden = e.HasTag(GameTag.PREMIUM),
				Taunt = e.HasTag(GameTag.TAUNT),
				DivineShield = e.HasTag(GameTag.DIVINE_SHIELD),
				Poisonous = e.HasTag(GameTag.POISONOUS),
				Venomous = e.HasTag(GameTag.VENOMOUS),
				Reborn = e.HasTag(GameTag.REBORN),
				Windfury = e.HasTag(GameTag.WINDFURY) || e.HasTag(GameTag.MEGA_WINDFURY),
				Stealth = e.HasTag(GameTag.STEALTH),
				Deathrattle = e.HasTag(GameTag.DEATHRATTLE),
				Frozen = e.HasTag(GameTag.FROZEN),
			};
		}

		private static void AddRace(List<string> races, Race? race)
		{
			if(race == null || race == Race.INVALID || race == Race.BLANK)
				return;
			var name = race.Value.ToString();
			if(!races.Contains(name))
				races.Add(name);
		}
	}
}
